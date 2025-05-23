#!/bin/bash
# Copyright 2024 ByteDance and/or its affiliates.
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
python experiments/benchmark/run_ruler.py \
    --model '/root/autodl-tmp/cache/model/Llama-3.1-8B-Instruct' \
    --task ruler \
    --chat \
    --attention headwise \
    --cfg "block_size=128,prefill_max_budget=2048,gamma=0.95,block_size=64" \
    --tag headwise\