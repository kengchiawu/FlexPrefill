# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 ByteDance and/or its affiliates.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)

from flex_prefill.ops.headwise_attention import init_headwise_attention

def llama_attn_forward_Headwise(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    softmax_scale: Optional[float] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    init_headwise_attention(self, num_hidden_layers=32)
    # 创建CUDA事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # 同步并记录开始时间
    torch.cuda.synchronize()
    start_event.record()
    
    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        raise ValueError("Only support self.config.pretraining_tp = 1 now.")
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    if position_embeddings is None: # Add to fix 4.44.2
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            self.is_prefill = True
            self.pre_len = kv_seq_len
            # key_state.shape = [batch_size, num_heads, seq_len, head_dim]
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            
            # raise ValueError(f"head_wise 待修改,添加triton版本")
            # prefill_mask.shape = [batch_size, num_heads, 1, kv_len]
            # head_wise_budget.shape = [bsz, self.num_heads]
            # prefill_mask = prefill_mask.repeat(1,1,q_len,1)
        else:
            self.kv_seq_len += q_len
            self.is_prefill = False
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            # attn_output = self.headwise_attention.headwise_attention_computation_decode()
    if self.is_prefill & self.layer_idx>1:
        attn_output = self.headwise_attention.headwise_attention_computaion_prefill(
                query_states,
                key_states,
                value_states,
                # gamma = self.config.headwise_gamma,
            )
    else:
        attn_output = flash_attn_func(
                query_states, key_states, value_states, softmax_scale=softmax_scale, causal=True
            )
        '''
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        #[batch_size, num_heads, q_len, kv_len]
        
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states) #value.shape = [bs, n_heads, kv_len, head_dim]
        '''
    
    '''
    if self.is_prefill and self.layer_idx>2:# 此处注意第1、2层不适用压缩方法
        # raise ValueError(f"head_wise 待修改,添加triton版本")
        # attn_weights = attn_weights * prefill_mask
        # attn_weights = attn_weights.masked_fill(~prefill_mask, float('-inf'))
        sotred_indices = torch.argsort(attn_weights,dim=-1,descending=True)
        arange = torch.arange(kv_seq_len,device=attn_weights.device).view(1,1,1,-1).expand(bsz,self.num_heads,q_len,-1)
        budget_expanded = head_wise_budget.unsqueeze(-1).repeat(1,1,q_len).unsqueeze(-1)
        # raise ValueError(f"arange:{arange.shape}\nbudget_expanded:{budget_expanded.shape}")
        mask = arange < budget_expanded
        prefill_mask = torch.zeros_like(attn_weights,dtype=torch.bool)
        prefill_mask.scatter_(-1,sotred_indices,mask)
        attn_weights = attn_weights.masked_fill(~prefill_mask, float('-inf'))
    '''
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    # 记录结束时间
    end_event.record()
    torch.cuda.synchronize()
    
    # 存储层计算时间
    elapsed = start_event.elapsed_time(end_event)
    self.config.timer.append({
        'layer':self.layer_idx,
        'time_ms':elapsed,
        'step':(self.kv_seq_len - self.pre_len)+1
    })
    return attn_output, attn_weights, past_key_value
    


