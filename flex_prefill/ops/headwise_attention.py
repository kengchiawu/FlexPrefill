import math
import torch
import warnings
import triton
import triton.language as tl
from typing import List, Optional, Tuple, Union
from transformers.utils import is_flash_attn_2_available
from einops import rearrange
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

def score_cover_topk(x: torch.Tensor, score: float):
    cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
    topk = torch.sum(cumsum_x <= score, dim=-1) + 1
    # torch.save(x,f"/homeB/youkangqi/SCOPE/results/llama-3.1-8b-instruct_2048_eager/attn_score/head_wise/layer_0_prelen_2598.pt")
    # raise ValueError(f"{topk.shape}")
    return topk

@triton.jit
def prefill_kernel(
    q_ptr,  # Q: b x h x n x d
    k_ptr,  # K: b x h x n x d
    v_ptr,  # V: b x h x n x d
    o_ptr,
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    Q_LEN,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    # softmax_scale
    softmax_scale,
    # causal
    causal,
    # gqa
    gqa_interleave,
    # stride
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
):
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)
    # full attention or causal attention
    lo = 0
    if causal:
        hi = min(K_LEN, (pid_q + 1) * BLOCK_SIZE_Q)
    else:
        hi = K_LEN
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(1,), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        if causal:
            qk += tl.where(off_m[:, None] >= (i + off_n)[None, :], 0, float("-inf"))
        else:
            qk += tl.where((off_n < K_LEN - i)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    # final scale
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(tl.float16), boundary_check=(0,))

def triton_flash_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
):
    batch_size, num_q_heads, q_len, head_dim = q.shape
    batch_size, num_kv_heads, k_len, head_dim = k.shape
    assert v.shape == k.shape
    #assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    # gqa
    assert num_q_heads % num_kv_heads == 0
    num_share_q_heads = num_q_heads // num_kv_heads
    # softmax_scale needs to be multiplied by math.log2(math.e)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # output tensor
    o = torch.zeros_like(q)

    grid = lambda META: (
        triton.cdiv(q_len, META["BLOCK_SIZE_Q"]),
        batch_size * num_q_heads,
    )
    # set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
    num_warps = 8
    BLOCK_SIZE_Q = min(
        128, max(16, triton.next_power_of_2(q_len))
    )  # min block size of tl.dot: 16
    BLOCK_SIZE_K = 128
    prefill_kernel[grid](
        q,
        k,
        v,
        o,
        batch_size,
        num_q_heads,
        num_kv_heads,
        num_share_q_heads,
        q_len,
        k_len,
        head_dim,
        softmax_scale,
        causal,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=2,
    )
    return o


@triton.jit
def decode_kernel(
    q_ptr,  # Q: b x h x 1 x d
    k_ptr,  # K: b x h x n x d
    v_ptr,  # V: b x h x n x d
    acco_ptr,  # acc_o: b x c x h x d
    lse_ptr,  # lse: b x c x h
    mi_ptr,  # mi: b x c x h
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    K_LEN,
    NUM_CHUNKS,
    HEAD_DIM: tl.constexpr,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave,
    # stride
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lc,
    stride_lh,
    stride_mb,
    stride_mc,
    stride_mh,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    CHUNK_SIZE_K: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE_K % BLOCK_SIZE_K == 0)
    # get batch id and head id
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    pid_c = tl.program_id(1)
    # init qkv pointer
    q_ptrs = (
        q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + tl.arange(0, HEAD_DIM) * stride_qd
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, pid_c * CHUNK_SIZE_K),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(pid_c * CHUNK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs)
    # init statistics
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((HEAD_DIM,), 0, dtype=tl.float32)
    # full attention
    lo = pid_c * CHUNK_SIZE_K
    hi = min(K_LEN, (pid_c + 1) * CHUNK_SIZE_K)
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(1,), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
        qk += tl.where((off_n < hi - i), 0, float("-inf"))
        qk += tl.sum(q[:, None] * k, axis=0) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.math.exp2(qk - m_ij)
        l_ij = tl.sum(p, axis=0)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale
        # load v and update acc_o
        v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.sum(p[:, None] * v, axis=0)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    # no final scale, do scale after all chunks are computed
    # acc_o = acc_o * tl.math.exp2(m_i - lse_i)
    # save lse and mi
    lse_ptr = (
        lse_ptr
        + pid_b * stride_lb
        + pid_h * stride_lh
        + (pid_c + tl.arange(0, 1)) * stride_lc
    )
    tl.store(lse_ptr, lse_i)
    mi_ptr = (
        mi_ptr
        + pid_b * stride_mb
        + pid_h * stride_mh
        + (pid_c + tl.arange(0, 1)) * stride_mc
    )
    tl.store(mi_ptr, m_i)
    # save chunk output
    off_d = tl.arange(0, HEAD_DIM)
    o_ptrs = (
        acco_ptr
        + pid_b * stride_ob
        + pid_c * stride_oc
        + pid_h * stride_oh
        + off_d * stride_od
    )
    tl.store(o_ptrs, acc_o)

@triton.jit
def rescale_kernel(
    acco_ptr,  # acc_o: b x c x h x d
    o_ptr,  # o: b x 1 x h x d
    lse_ptr,  # lse: b x c x h
    mi_ptr,  # mi: b x c x h
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_CHUNKS,
    HEAD_DIM: tl.constexpr,
    # stride
    stride_ab,
    stride_ac,
    stride_ah,
    stride_ad,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lc,
    stride_lh,
    stride_mb,
    stride_mc,
    stride_mh,
    # META parameters
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # get batch id and head id
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    # ptrs
    off_chunks = tl.arange(0, BLOCK_SIZE_C)
    mi_ptrs = mi_ptr + pid_b * stride_mb + pid_h * stride_mh + off_chunks * stride_mc
    lse_ptrs = lse_ptr + pid_b * stride_lb + pid_h * stride_lh + off_chunks * stride_lc
    acco_ptrs = tl.make_block_ptr(
        base=acco_ptr + pid_b * stride_ab + pid_h * stride_ah,
        shape=(NUM_CHUNKS, HEAD_DIM),
        strides=(stride_ac, stride_ad),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_C, BLOCK_SIZE_D),
        order=(1, 0),
    )
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(1, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(0, 0),
        block_shape=(1, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # load mi and lse
    mi = tl.load(mi_ptrs, mask=off_chunks < NUM_CHUNKS, other=float("-inf"))
    lse = tl.load(lse_ptrs, mask=off_chunks < NUM_CHUNKS, other=float("-inf"))
    # get scale factor
    m = tl.max(mi, axis=0)
    scale = tl.math.exp2(mi - m) / tl.sum(tl.math.exp2(lse - m), axis=0)
    # reduce
    o = tl.full((HEAD_DIM,), 0, dtype=tl.float32)
    for i in range(0, HEAD_DIM, BLOCK_SIZE_D):
        i = tl.multiple_of(i, BLOCK_SIZE_D)
        # rescale and reduce
        acco = tl.load(acco_ptrs, boundary_check=(0, 1), padding_option="zero")
        acco = tl.sum(acco * scale[:, None], axis=0)[None, :]
        # save
        tl.store(o_ptrs, acco.to(tl.float16), boundary_check=(0, 1))
        # update ptrs
        acco_ptrs = tl.advance(acco_ptrs, (0, BLOCK_SIZE_D))
        o_ptrs = tl.advance(o_ptrs, (0, BLOCK_SIZE_D))

def triton_flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
):
    batch_size, num_q_heads, q_len, head_dim = q.shape
    batch_size, num_kv_heads, k_len, head_dim = k.shape
    assert q_len == 1
    assert v.shape == k.shape
    #assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    # softmax_scale needs to be multiplied by math.log2(math.e)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # gqa
    assert num_q_heads % num_kv_heads == 0
    num_share_q_heads = num_q_heads // num_kv_heads
    # grid
    grid = lambda META: (
        batch_size * num_q_heads,  # batch & head
        triton.cdiv(k_len, META["CHUNK_SIZE_K"]),  # k chunks
    )
    # set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
    num_warps = 8
    BLOCK_SIZE_K = 128
    CHUNK_SIZE_K = 4096
    # chunk output and chunk lse and chunk
    num_chunks = triton.cdiv(k_len, CHUNK_SIZE_K)
    lse = torch.empty(
        batch_size, num_chunks, num_q_heads, dtype=torch.float32, device=q.device
    )
    mi = torch.empty(
        batch_size, num_chunks, num_q_heads, dtype=torch.float32, device=q.device
    )
    acc_o = torch.empty(
        batch_size,
        num_chunks,
        num_q_heads,
        head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    # launch kernel
    decode_kernel[grid](
        q,
        k,
        v,
        acc_o,
        lse,
        mi,
        batch_size,
        num_q_heads,
        num_kv_heads,
        num_share_q_heads,
        k_len,
        num_chunks,
        head_dim,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        acc_o.stride(0),
        acc_o.stride(1),
        acc_o.stride(2),
        acc_o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        mi.stride(0),
        mi.stride(1),
        mi.stride(2),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        CHUNK_SIZE_K=CHUNK_SIZE_K,
        num_warps=num_warps,
        num_stages=2,
    )
    # rescale
    o = torch.empty(
        batch_size,
        1,
        num_q_heads,
        head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    # grid
    grid = lambda META: (batch_size * num_q_heads,)  # batch & head
    # set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
    num_warps = 4 if head_dim <= 64 else 8
    BLOCK_SIZE_C = triton.next_power_of_2(num_chunks)
    BLOCK_SIZE_D = min(head_dim, 128 * 128 // BLOCK_SIZE_C)
    # launch kernel
    rescale_kernel[grid](
        acc_o,
        o,
        lse,
        mi,
        batch_size,
        num_q_heads,
        num_chunks,
        head_dim,
        acc_o.stride(0),
        acc_o.stride(1),
        acc_o.stride(2),
        acc_o.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        mi.stride(0),
        mi.stride(1),
        mi.stride(2),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        num_warps=num_warps,
        num_stages=2,
    )
    return o


def get_headwise_block_idx_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 16,
        prefill_max_blocks: int = 128,
        prefill_min_blocks: int = 4,
        gamma: float = 0.95,
        gqa_interleave: bool = False,
):
    """
    Based on head_wise_budget, select important token in block level:

    Args:
        self: class Head_Wise_Attention itself, we can store head_wise_budget in it.
        q (torch.Tensor): Query states, shape [batch_size, num_heads, seq_lens, head_dim]
        k (torch.Tensor): Key states, same as query
        v (torch.Tensor): Value states, same as query
        gamma(float):筛选阈值,默认为0.95
        
        block_size (int): Block size, 
        
    Returns:
        head_wise_budget (torch.Tensor):  shape [batch_size, num_heads], 
        block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num], which is the index of the flattened block grid.
            For example, in a 4x4 block grid, if you want to activate 5 blocks: (0,0), (1,1), (2,0), (3,1), (3,2), the index will be: [0, 5, 8, 13, 14]
    """
    batch_size, num_heads, q_len, head_dim = query.shape
    batch_size, num_heads, kv_len, head_dim = key.shape
    num_share_q_heads = num_heads // key.shape[2]
    assert q_len == kv_len , "only support in prefill phase"
    assert batch_size == 1, "only support for batch_size=1"
    last_q = query[:,  :,-block_size:, :]
    qk = torch.matmul(last_q,key.transpose(2,3))/math.sqrt(head_dim)

    '''block_size暂时默认为1,不考虑causal_mask'''
    alloc_causal_mask = torch.arange(0, block_size, device=last_q.device)
    alloc_causal_mask = alloc_causal_mask[:, None] >= alloc_causal_mask[None, :]
    alloc_causal_mask = alloc_causal_mask[None, None, ...]
    qk[..., -block_size:].masked_fill_(
        ~alloc_causal_mask[..., :block_size, :block_size], float("-inf")
    )
    
    # softmax，上采样到fp32
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    # qk = rearrange(qk, "b h g i j -> b (h g) i j")
    # qk.shape = [batch_size, num_heads, block_size, key_len]
    vertical = qk.mean(-2)
    head_wise_budget = score_cover_topk(vertical,gamma)# shape = [bs, nh]
    # raise ValueError(f"\nnum_tokens:{num_tokens.shape}\nqk:{qk.shape}")
    # 接下来是生成head_wise_budget，并思考如何进行head_wise_spare的分配
    # 初步想法是生成一个mask矩阵，对注意力分数排序后的indicates矩阵进行掩码
    # 该矩阵和head_wise_budget一样，生成一次就行
    num_blocks = math.ceil(q_len / block_size)
    prefill_max_blocks = min(prefill_max_blocks, num_blocks)
    max_budget = torch.max(head_wise_budget)
    head_wise_budget = torch.div(head_wise_budget,max_budget) * block_size * prefill_max_blocks
    budget_blocks = torch.div(head_wise_budget, block_size, rounding_mode='floor')+1
    budget_blocks = torch.clamp(budget_blocks,min=prefill_min_blocks,max=prefill_max_blocks)
    #self.budget_blocks = budget_blocks

    num_blocks = math.ceil(kv_len/block_size)
    pad_len = num_blocks * block_size - kv_len
    avg_k = (
        torch.nn.functional.pad(key,(0,0,pad_len,0),value=0)
        .view(batch_size,num_heads,num_blocks,block_size,head_dim)
        .mean(-2)
    )
    avg_k[:,:,-1,:] = avg_k[:,:,-1,:] * block_size / (block_size - pad_len)
    ''''''
    avg_q = (
        torch.nn.functional.pad(query,(0,0,pad_len,0),value=0)
        .view(batch_size,num_heads,num_blocks,block_size,head_dim)
        .mean(-2)
    )
    avg_q[:,:,-1,:] = avg_q[:,:,-1,:] * block_size / (block_size - pad_len)
    
    block_causal_mask = torch.tril(
                torch.ones((num_blocks, num_blocks), device=query.device, dtype=torch.bool)
            ).repeat(batch_size,num_heads,1,1)
    block_attn = torch.einsum(
        "bhid, bhjd -> bhij", avg_q / math.sqrt(head_dim), avg_k
    ).masked_fill_(~block_causal_mask, float("-inf"))
    block_attn = torch.softmax(block_attn, dim=-1, dtype=torch.float32)
    # block_attn.shape = [bs, nh, q_blocks, k_blocks]
    block_attn_indices = torch.argsort(block_attn, dim=-1, descending=True)
    # block_attn_indices.shape = [bs, nh, q_blocks, k_blocks]
    arange = torch.arange(num_blocks,device=block_attn.device).view(1,1,1,num_blocks)
    budget_block_expand = budget_blocks[:,:,None,None].expand(-1,-1,num_blocks,-1) # [batch_size, num_heads, num_blocks, 1]
    block_indices_mask = arange < budget_block_expand # [batch_size, num_heads, num_blocks, num_blocks]
    select_block_idx = block_attn_indices * block_indices_mask
    # select_block_idx = torch.unique(select_block_idx,dim=-1)
    # raise ValueError(f"{block_indices_mask[:,:,-1,:].sum(-1)}")
    # raise ValueError(f"select_block_idx[0][0][0]: {select_block_idx[0][0][0]} \n select_block_idx[0][1][15]: {select_block_idx[0][1][15]}\n select_block_idx[0][2][27]: {select_block_idx[0][2][27]}")
    return select_block_idx,budget_blocks

    '''
    # head_wise_budget[head_wise_budget<prefill_min_budget] = prefill_min_budget
    attn_score_3d = qk.mean(dim=-2) # [batch, num_heads, kv_len]
    
    # 对每个头的分数进行降序排序，获取排序后的索引
    sorted_indices = torch.argsort(attn_score_3d, dim=-1, descending=True)  # [batch, num_heads, kv_len]
    
    # 生成位置索引并与预算比较，生成前k的布尔掩码
    arange = torch.arange(kv_len, device=qk.device).view(1, 1, -1).expand(batch_size, num_heads, -1)
    budget_expanded = head_wise_budget.unsqueeze(-1)  # [batch, num_heads, 1]
    mask = arange < budget_expanded  # [batch, num_heads, kv_len]

    # 将排序后的掩码映射回原始位置
    final_mask = torch.zeros_like(attn_score_3d, dtype=torch.bool)
    final_mask.scatter_(-1, sorted_indices, mask)

    # 扩展掩码形状并应用到原始注意力分数
    final_mask = final_mask.unsqueeze(2) # [batch, num_heads, 1, kv_len]
    selected_scores = qk * final_mask
    # raise ValueError(f"\nfinal_mask:{final_mask.shape}")
    return head_wise_budget
    '''


@triton.jit
def count_kernel(
    x_ptr,
    y_ptr,
    k,
    r,
    stride_xb,
    stride_xh,
    stride_xk,
    stride_yb,
    stride_yh,
    stride_yr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    # load x
    x_ptr = x_ptr + pid_b * stride_xb + pid_h * stride_xh
    off_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + off_k * stride_xk
    y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)
    for i in range(0, k, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, off_k < k - i, -1)
        x = x // r
        x = tl.where(off_k < k - i, x, -1)
        # count
        # maybe triton bug: when BLOCK_SIZE_R == r, the count of values ​​in bin [r-1, r) will be wrong
        y += tl.histogram(x, BLOCK_SIZE_R)
        # move ptr
        x_ptrs = x_ptrs + BLOCK_SIZE_K * stride_xk
    # cumsum
    y = tl.cumsum(y, axis=0)
    # store result
    y_ptr = y_ptr + pid_b * stride_yb + pid_h * stride_yh + stride_yr
    off_r = tl.arange(0, BLOCK_SIZE_R)
    tl.store(y_ptr + off_r * stride_yr, y, off_r < r)

def triton_column_count_cumsum(x: torch.Tensor, num_columns: int) -> torch.Tensor:
    """count columns of each row for a given index tensor, then do cumsum

    Args:
        x (torch.Tensor): block index in a flatten 2d grid, shape [batch_size, num_heads, activated_block_num]
        num_colums (int): number of columns in the grid

    Returns:
        torch.Tensor: cumsum of columns num in each row, shape [batch_size, num_heads, num_rows + 1 ]
            For example, in a 4x4 block grid, activated blocks have index [0, 5, 8, 9, 13, 14], number of blocks in each row is [1, 1, 2, 2],
            this function will return cumsum tensor [0, 1, 2, 4, 6]
    """
    x = x.to(torch.int32)
    b, h, k = x.shape
    r = num_columns
    # torch implementation:
    # y = torch.zeros(b,h,r*r,dtype=x.dtype,device=x.device)
    # y[torch.arange(b,device=x.device)[:,None,None],torch.arange(h,device=x.device)[None,:,None],torch.where(x<r*r,x,0)]=1
    # y = torch.nn.functional.pad(torch.cumsum(y.view(b,h,r,r).sum(-1),-1),(1,0),value=0).to(torch.int32)
    block_size_k = min(triton.next_power_of_2(k), 4096)
    # plus r by 1 to avoid tl.histogram bug
    block_size_r = triton.next_power_of_2(r + 2)
    y = torch.zeros(b, h, r + 1, device=x.device, dtype=torch.int32)
    count_kernel[(b, h)](
        x,
        y,
        k,
        r,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        block_size_k,
        block_size_r,
    )
    return y


def torch_column_count_cumsum(x: torch.Tensor, num_columns: int) -> torch.Tensor:
    """count columns of each row for a given index tensor, then do cumsum

    Args:
        x (torch.Tensor): block index in a flatten 2d grid, shape [batch_size, num_heads, activated_block_num]
        num_colums (int): number of columns in the grid

    Returns:
        torch.Tensor: cumsum of columns num in each row, shape [batch_size, num_heads, num_rows + 1 ]
            For example, in a 4x4 block grid, activated blocks have index [0, 5, 8, 9, 13, 14], number of blocks in each row is [1, 1, 2, 2],
            this function will return cumsum tensor [0, 1, 2, 4, 6]
    """
    x = x.to(torch.int64)
    batch_size, num_heads, k = x.shape
    y = torch.zeros(
        batch_size, num_heads, num_columns + 1, dtype=torch.int32, device=x.device
    )
    mask = torch.zeros(
        (num_columns + 2) * num_columns, dtype=torch.bool, device=x.device
    )
    for b in range(batch_size):
        for h in range(num_heads):
            mask = mask.view(-1)
            mask.zero_()
            mask.index_fill_(dim=-1, index=x[b, h].view(-1), value=1)
            y[b, h, 1:] = (
                mask.view(num_columns + 2, num_columns)[:-2,].sum(-1).cumsum(-1)
            )
    return y



@triton.jit
def block_wise_prefill_attention_kernel(
    q_ptr,  # shape: [batch_size, seq_len, num_heads, head_dim]
    k_ptr,
    v_ptr,
    o_ptr,
    block_idx_ptr,  # shape: [batch_size, num_heads, num_all_block]
    idx_bin_ptr,  # shape: [batch_size, num_heads, seq_len / block_size + 1]
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    Q_LEN,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCK,
    grid_offset,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave: tl.constexpr,
    # stride
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    stride_bb,
    stride_bh,
    stride_bt,
    stride_ib,
    stride_ih,
    stride_it,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
):
    tl.static_assert(BLOCK_SIZE_Q == BLOCK_SIZE_K)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    pid_q = tl.program_id(2)
    # get column index bin
    idx_bin_ptr = idx_bin_ptr + pid_b * stride_ib + pid_h * stride_ih
    bin_start = tl.load(idx_bin_ptr + pid_q * stride_it)
    bin_end = tl.load(idx_bin_ptr + (pid_q + 1) * stride_it)
    num_active_block = bin_end - bin_start
    # get column block index ptr
    block_idx_ptr = (
        block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh + bin_start * stride_bt
    )
    # init qkv ptrs
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q - grid_offset
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)
    # flash attention
    for i in range(0, num_active_block):
        # get current block start index
        c = tl.load(block_idx_ptr).to(tl.int32) % NUM_BLOCK * BLOCK_SIZE_K - grid_offset
        block_idx_ptr = block_idx_ptr + stride_bt
        # load k
        k = tl.load(
            tl.advance(k_ptrs, (0, c)), boundary_check=(1,), padding_option="zero"
        )
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where((c + off_n)[None, :] >= 0, 0, float("-inf"))
        qk += tl.where(off_m[:, None] >= (c + off_n)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(
            tl.advance(v_ptrs, (c, 0)), boundary_check=(0,), padding_option="zero"
        )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
    # final scale
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(tl.float16), boundary_check=(0,))

def triton_block_wise_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: Union[torch.Tensor, List[List[torch.Tensor]]],
    block_size: int,
    grid_offset: int = 0,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

    Args:
        q (torch.Tensor): Query states, shape [batch_size, seq_lens, num_heads, head_dim]
        k (torch.Tensor): Key states, same as query
        v (torch.Tensor): Value states, same as query
        block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num], which is the index of the flattened block grid.
            For example, in a 4x4 block grid, if you want to activate 5 blocks: (0,0), (1,1), (2,0), (3,1), (3,2), the index will be: [0, 5, 8, 13, 14]
        block_size (int): Block size, only support 16, 32, 64 and 128.
        grid_offset (int): Move the grid that divides the block to the lower left corner by grid_offset, default to 0.
        softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
        gqa_interleave (bool): use interleave mode of gqa, default to False.

    Returns:
        torch.Tensor: Attention output, shape [batch_size, seq_lens, num_heads, head_dim]
    """
    batch_size, num_q_heads, q_len, head_dim = q.shape
    batch_size, num_kv_heads, k_len, head_dim = k.shape
    #assert q.dtype == torch.bfloat16
    assert q_len == k_len
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    assert block_size in {
        16,
        32,
        64,
        128,
    }, "only support block size in {16, 32, 64, 128}"
    total_q_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        q_len - grid_offset, block_size
    )
    total_k_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        k_len - grid_offset, block_size
    )
    # pad block_idx if get list[list[tensor]]
    if not isinstance(block_idx, torch.Tensor):
        assert (
            isinstance(block_idx, list)
            and isinstance(block_idx[0], list)
            and isinstance(block_idx[0][0], torch.Tensor)
        )
        assert len(block_idx) == batch_size and len(block_idx[0]) == num_q_heads
        block_idx = [item.view(-1, 1) for sublist in block_idx for item in sublist]
        block_idx = torch.nn.utils.rnn.pad_sequence(
            block_idx,
            batch_first=True,
            padding_value=total_k_blocks * (total_k_blocks + 1),
            # padding_value=0,
        )
        block_idx = block_idx.view(batch_size, num_q_heads, -1)
    batch_size, num_q_heads, num_block = block_idx.shape
    assert q_len == k_len
    assert num_block <= total_q_blocks * (total_q_blocks + 1) // 2
    # gqa
    assert num_q_heads % num_kv_heads == 0
    num_share_q_heads = num_q_heads // num_kv_heads
    # softmax_scale
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # sort idx and get block index bins
    block_idx = block_idx.sort(-1).values
    if triton.__version__ == "3.0.0":
        idx_bins = triton_column_count_cumsum(block_idx, total_k_blocks)
    else:
        warnings.warn("triton version 3.0.0 is required for faster attention")
        idx_bins = torch_column_count_cumsum(block_idx, total_k_blocks)
    # launch attention kernel
    o = torch.empty_like(q)
    num_warps = 8
    num_stages = 3 if block_size >= 128 else 5
    block_wise_prefill_attention_kernel[(batch_size, num_q_heads, total_q_blocks)](
        q,
        k,
        v,
        o,
        block_idx,
        idx_bins,
        batch_size,
        num_q_heads,
        num_kv_heads,
        num_share_q_heads,
        q_len,
        k_len,
        head_dim,
        total_q_blocks,
        grid_offset,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        idx_bins.stride(0),
        idx_bins.stride(1),
        idx_bins.stride(2),
        BLOCK_SIZE_Q=block_size,
        BLOCK_SIZE_K=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o

class Head_Wise_Attention():
    prefill_length = 0
    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0
    
    def __init__(
            self,
            prefill_max_budget = 2048,
            prefill_min_budget = 128,
            decode_metric = 'None',
            decode_budget = 1024,
            block_size = 16,
            num_hidden_layers = 32,
            gamma = 0.95,
    ):
        self.prefill_max_budget = prefill_max_budget
        self.prefill_min_budget = prefill_min_budget
        self.decode_metric = decode_metric
        self.decode_budget = decode_budget
        self.block_size = block_size
        self.num_hidden_layers = num_hidden_layers
        self.gamma = gamma
        
    def reset(
            self,
            prefill_max_budget = 1024,
            prefill_min_budget = 128,
            decode_metric = 'None',
            decode_budget = 1024,
            block_size = 16,
            num_hidden_layers = 32,
    ):
        self.prefill_max_budget = prefill_max_budget
        self.prefill_min_budget = prefill_min_budget
        self.decode_metric = decode_metric
        self.decode_budget = decode_budget
        self.block_size = block_size
        self.num_hidden_layers = num_hidden_layers

    def headwise_attention_computaion_prefill(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            # is_prefill: bool,
            softmax_scale: Optional[float] = None,
            return_computational_ratio: Optional[bool] = False,
    )-> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        batch_size, q_len, num_q_heads, head_dim = q.shape
        batch_size, kv_len, num_k_heads, head_dim = k.shape
        assert batch_size == 1, "only support batch size 1 for now"
        assert q_len > 1, "support for first token generation"
        #if self.layer_idx < 2:
        #    q
        self.pre_len = kv_len
        head_wise_block_idx,self.budget_blocks = get_headwise_block_idx_prefill(
            self,
            q,
            k,
            block_size = self.block_size,
            prefill_max_blocks=self.prefill_max_budget/self.block_size,
            prefill_min_blocks=self.prefill_min_budget/self.block_size,
            gamma=self.gamma,
        )
        #if not isinstance(head_wise_block_idx, torch.Tensor):
        #raise ValueError(f"head_wise_block_idx.shape: {head_wise_block_idx.shape}")
        attn = triton_block_wise_prefill_attention(
            q,
            k,
            v,
            block_idx=head_wise_block_idx,
            block_size=self.block_size,
        )
        

        return attn # [batch_size, num_heads, max_prefill/block_size+1 ]

    def headwise_attention_computaion_flash(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            causal: bool = True,
            softmax_scale: Optional[float] = None,
            gqa_interleave: bool = False,
            attention_mask: torch.Tensor = None,
    ):
        batch_size, num_heads, q_len, head_dim = q.shape
        batch_size, num_heads, k_len, head_dim = k.shape
        assert v.shape == k.shape
        #assert q.dtype == torch.bfloat16, f"only support dtype bfloat16,but q.dtype={q.dtype}"
        assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        #[batch_size, num_heads, q_len, kv_len]
        
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : k.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        #attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v) #value.shape = [bs, n_heads, kv_len, head_dim]
        return attn_output
        if q_len > 1:
            return triton_flash_prefill(q,k,v,softmax_scale,gqa_interleave)
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            #[batch_size, num_heads, q_len, kv_len]
            
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : k.shape[-2]]
                attn_weights = attn_weights + causal_mask
            
            # upcast attention to fp32
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            #attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, v) #value.shape = [bs, n_heads, kv_len, head_dim]
            return attn_output
        else:
            return triton_flash_decode(q, k, v, softmax_scale, gqa_interleave).transpose(1,2)

def init_headwise_attention(self, num_hidden_layers):
    if not hasattr(self, "headwise_attention"):
        if not hasattr(self.config, 'decode_metric'):
            self.config.decode_metric = 'None'
        if not hasattr(self.config, 'prefill_max_budget'):
            self.config.prefill_max_budget = 1024
        if not hasattr(self.config, 'prefill_min_budget'):
            self.config.prefill_min_budget = 128
        if not hasattr(self.config, 'decode_budget'):
            self.config.decode_budget = 512
        if not hasattr(self.config, 'block_size'):
            self.config.block_size = 16
        if not hasattr(self.config, 'gamma'):
            self.config.gamma = 0.95
    
    self.headwise_attention = Head_Wise_Attention(
        prefill_max_budget = self.config.prefill_max_budget,
        prefill_min_budget = self.config.prefill_min_budget,
        decode_metric = self.config.decode_metric,
        decode_budget = self.config.decode_budget,
        block_size = self.config.block_size,
        num_hidden_layers = num_hidden_layers,
        gamma = self.config.gamma,
    )