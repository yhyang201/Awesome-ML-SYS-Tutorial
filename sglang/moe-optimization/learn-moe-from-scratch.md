
# 从0开始学习MOE/MOE推理优化

为了理解方便，这里从 Qwen3MOE 开始介绍，后续补充 Deepseek V3

## Overview

1. Qwen3MoE native 实现: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
2. FusedMoE优化: https://github.com/sgl-project/sglang/blob/4db463b1ad6edcd6b8cd500be377f65ff8e3b419/python/sglang/srt/models/qwen3_moe.py
3. EP Moe: https://github.com/sgl-project/sglang/commit/e330f2b86cd23f1acec113378aebd7bee268830b
4. DeepEP: https://github.com/sgl-project/sglang/pull/6120
