
# 从0开始学习MOE/MOE推理优化

为了理解方便，这里从 Qwen3MOE 开始介绍，后续补充 Deepseek V3

## Overview

1. Qwen3MoE native 实现: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
2. FusedMoE优化: https://github.com/sgl-project/sglang/blob/4db463b1ad6edcd6b8cd500be377f65ff8e3b419/python/sglang/srt/models/qwen3_moe.py
3. EP Moe: https://github.com/sgl-project/sglang/commit/e330f2b86cd23f1acec113378aebd7bee268830b
4. DeepEP: https://github.com/sgl-project/sglang/pull/6120


## Qwen3MoE Native

我们先来看 `Qwen3MoeDecoderLayer`，其定义如下：
```
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```
可以看到
如果满足 `if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):`
条件，则使用 `Qwen3MoeSparseMoeBlock`。

在实际的 Qwen3MOE 30B，235B模型中，config 中有
```
"decoder_sparse_step": 1,
"mlp_only_layers": [],
"num_experts": 128,
```

因此实际模型每一层都使用了 `Qwen3MoeSparseMoeBlock`。
`forward` 没有什么特殊之处，直接调用 self.mlp 进行前向传播。


进一步我们来看 `Qwen3MoeSparseMoeBlock` ，其定义为：
```
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
```

`num_experts` 表示专家数量，实际为 128

`top_k`  表示每个 token 需要 `top_k` 个专家处理，实际为 8

`norm_topk_prob` 参数控制了在选出 top_k 个最相关的专家后，是否将它们的路由权重归一化。实际为 True

`self.gate` 帮助我们判断token应该送去哪个expert了。在别的MoE架构中，Gate有时也被称为Router（路由）。Gate的尺寸大小为(M, E)，其中E表示expert的数量。
输入数据(S, H)过Gate(H, E)后，得到prob数据(S, E)，它的含义是：每个token去向每个expert的概率。

`self.experts` num_experts 个专家本身，每个 experts 都是一个 Qwen3MoeMLP



接下来我们来学习 Forward 过程, 讲解在代码注释中：

```
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        ## 将不同 batch 的 sequences 展平

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

      
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        ## routing_weights: (batch_size * sequence_length, self.top_k)
        ## selected_experts: (batch_size * sequence_length, self.top_k)


        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # 归一化
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        ## final_hidden_states 汇集了每个输入token经过其被选中的各个专家处理并加权后的最终输出结果。

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        ## (num_experts, top_k, batch_size* sequence_length)

        # expert_mask 通过对每个token选中的top_k专家索引进行独热编码并重排维度，创建了一个形状为 (总专家数, top_k值, 总token数) 的布尔掩码.
        # expert_mask 表示，对于第 shape[2] 个 token来说，其第 shape[1] 个 expert，是否是 shape[0] 的专家。如果是 则为 True，否则为 False
        # NOTE: 这里维护了 top_k 的顺序，需要思考为什么需要维护这个顺序？

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            ## idx: 对于这些选择了当前 expert_idx 的token，当前这个 expert_idx 是它们各自 top_k 个选择中的第几个
            ## top_x: 哪些输入token（由 top_x 标识） 选择了当前这个特定的 expert_idx 作为它们 top_k 个被路由到的专家之一
            ## idx_, top_x 长度一致，是 expert_mask 中 True 的数量，假设其长度为 num_choose_this_expert

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            ## current_state: (num_choose_this_expert, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            ## routing_weights[top_x, idx, None] (num_choose_this_expert, 1)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
```

简而言之，就是循环迭代每一个 expert 进行计算。


## FusedMoE

FusedMoE 的核心思想是将原生实现中分散的、开销大的操作“融合”到少数几个（甚至一个）高度优化的自定义 GPU Kernel 中，以大幅提升性能。

FusedMoE:

`__init__`:  BF16版本使用 `UnquantizedFusedMoEMethod`

`forward`: 
```
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            correction_bias=self.correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states
```


UnquantizedFusedMoEMethod:


```
    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        if _is_hip and get_bool_env_var("SGLANG_AITER_MOE"):
            assert not no_combine, "unsupported"
            return ck_moe_2stages(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                activation=(
                    ActivationType.Silu if activation == "silu" else ActivationType.Gelu
                ),
            )
        else:
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace and not no_combine,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                no_combine=no_combine,
            )

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        return moe_forward_native(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function,
            correction_bias,
        )
```
注意，在 cuda graph capture 时，会执行：
```
def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
                setattr(sub, "is_torch_compile", False)
            else:
                # NOTE: Temporarily workaround MoE
                if "FusedMoE" in sub.__class__.__name__:
                    if num_tokens == 1:
                        # The performance of torch.compile on this layer is not always good when bs > 1,
                        # so we decide to only use torch.compile when bs =1
                        sub._forward_method = fused_moe_forward_native
                else:
                    sub._forward_method = sub.forward_native
                setattr(sub, "is_torch_compile", True)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)
```


## EP Moe

TODO 

ref: 

https://zhuanlan.zhihu.com/p/21251657579

https://zhuanlan.zhihu.com/p/17790182311






Reference: 
https://zhuanlan.zhihu.com/p/681154742

