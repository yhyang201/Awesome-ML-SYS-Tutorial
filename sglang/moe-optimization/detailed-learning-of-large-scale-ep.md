
原文地址：https://lmsys.org/blog/2025-05-05-large-scale-ep/

> DeepSeek 是一个广受好评的开源大型语言模型（LLM），以其强大的性能著称。然而，由于其规模庞大且采用了多头潜在注意力机制（Multi-head Latent Attention, MLA）和专家混合机制（Mixture of Experts, MoE）等独特架构，因此需要一个高级系统来实现高效的大规模推理服务。在本博客中，我们将介绍我们是如何通过 SGLang 匹配 DeepSeek 推理系统性能的。


![image](https://github.com/user-attachments/assets/084f50cf-8ec5-4c18-9dba-3d15aeafa1b3)


> 我们的实现如上图所示，**在 Atlas Cloud 上的 12 个节点上运行，每个节点配备 8 张 H100 GPU**。该系统采用了**预填-解码分离架构（prefill-decode disaggregation）**以及**大规模专家并行（EP）策略**，在处理**长度为 2000 的输入序列时**，**每个节点可实现每秒 52,300 个输入 token 和 22,300 个输出 token 的速度**。据我们所知，这是首个在大规模设置下几乎达到官方 DeepSeek 博客所述吞吐量的开源实现。

TODO: 查询一下官方速度

> 将该实现部署到本地时，**成本约为每 100 万个输出 token 花费 0.20 美元**，仅为官方 DeepSeek Chat API 成本的五分之一。与使用相同资源的**基础张量并行（vanilla tensor parallelism）相比，该优化策略最多可提升 5 倍的输出吞吐率**。

学习一下成本是如何计算的，以及vanilla tensor parallelism

> 本博客将深入介绍我们的并行设计、优化方法以及实验结果。我们的所有组件均为完全开源，便于他人探索并在此基础上进一步开发。实验复现的全部步骤和说明已在此提供。

## Highlight

> ✅ SGLang 现已支持预填-解码（Prefill-Decode, PD）分离以及大规模专家并行（EP），包括 DeepEP、DeepGEMM 和 EPLB 的全部功能。》
>
> ✅ 借助这些新特性，我们团队成功复现了 DeepSeek 的推理系统，在 12 个节点上运行，每个节点配备 8 张 H100 GPU。总体上，SGLang 实现了每个节点每秒 52,300 个输入 token 和 22,300 个输出 token 的吞吐率（基于 2000 个 token 的输入序列）。
>
> ✅ 本博客详细解释了我们方法的技术细节，重点介绍了在效率优化、峰值内存使用减少以及负载均衡方面的改进。性能分析结果显示，我们的实现几乎达到了官方 DeepSeek 报告中所述的性能水平。
> 
> ✅ 所有实验和代码已完全开源，供社区访问与进一步开发。

## Parallelism Design

> 高效的并行策略对于应对 DeepSeek 架构所带来的计算复杂性和内存需求至关重要。本节将概述我们在以下关键组件中的优化方法：注意力层、稠密前馈网络（FFN）、稀疏 FFN，以及语言模型（LM）头。每个组件都采用了专门定制的并行策略，以提升可扩展性、内存效率和整体性能。



### Attention Layers

> DeepSeek 采用多头潜在注意力机制（Multi-head Latent Attention, MLA）来有效建模输入序列中的复杂依赖关系。为优化这一机制，我们实现了 DP Attention——一种数据并行策略，它消除了设备间 KV 缓存的冗余，大幅降低了内存开销。该方法在 SGLang v0.4 中首次引入，并已扩展为**支持数据并行与张量并行的混合模式**，为高效处理小批量数据提供了更大的灵活性。

需要学习一下 DP Attention（代码级）

hybrid data and tensor parallelism 是什么？

### Dense FFNs


> 尽管仅使用三个稠密 FFN 层，DeepSeek-V3 的计算可能会显著增加峰值内存使用量，如果不加以管理，可能会导致系统崩溃。为了解决这个问题，我们采用了**数据并行（Data Parallelism, DP）**替代张量并行（Tensor Parallelism, TP），并利用以下优势：
>
> - Enhanced Scalability: **当中间维度为 18,432 时，高 TP 度数（例如 TP=32）会将数据分割成许多小单元（如 576 个单元），这无法被 128 整除，而 128 是现代 GPU（如 H100）的常见对齐边界。**这种不对齐会降低计算效率和内存利用率。DP 通过避免碎片化实现更可扩展的解决方案，确保跨设备的负载均衡分配。
>
> - Optimized Memory Efficiency: 传统上，TP 会随着工作节点数量增加而减少内存使用，但这种优势在 DP 注意力机制下会减弱。在纯 TP 设置中，单层 Transformer 模型的内存需求按以下公式随 DP 大小扩展：
>
> $$
> \text{Memory} = \frac{N_{\text{param}}}{\text{TP}} + (1 + k) N_{\text{hidden state}} \cdot \text{DP}
> $$
> 
> 其中，N_hidden_state = n_token × n_hidden_size 是每个设备（DP rank）上的隐藏状态大小，N_param = n_intermediate_size × n_hidden_size 是模型参数数量，k 是 CUDA Graph 复制带来的额外内存开销系数。
> 
> 假设 DP = TP，这个内存使用函数在：
> 
> TP = sqrt(N_param / ((1 + k) × N_hidden_state))
> 
> 时最小。DeepSeek-V3 使用 18,432 的中间维度。在预填阶段，通常禁用 CUDA Graph，此时 $k=0$。但每个设备上的 token 数量可能超过 2048，因此最佳 TP 度数为 3 或更少。在解码阶段，实际配置可能是每设备 128 个 token，设 
> $k=3$，此时最佳 TP 为 6。无论在哪个阶段，较低的 TP 度数可以减少每个设备的内存使用。因此，相比完全依赖 TP，DP 在扩展时可能更节省内存。
>
> - Minimized Communication Overhead: **在纯 TP（张量并行）模式下，每个 FFN（前馈网络）都需要执行两次 all-reduce 操作，带来大量通信开销。** 通过采用 DP（数据并行），我们将这一过程优化为：在前一个注意力层之后进行一次 reduce-scatter，在下一个层之前进行一次 all-gather，从而将通信成本减少了 50%。此外，当注意力机制也在纯 DP 下计算时，设备间的通信将被完全消除，显著提升整体效率。
>
>   DP 稠密 FFN 与 DP 注意力机制的结合在下方左侧的图中进行了展示。用户可以通过设置 --moe-dense-tp-size=1 来启用该功能。
>
>   ![image](https://github.com/user-attachments/assets/4ca264f5-aa6f-4101-80ec-163cf9d03ce2)
>
> 

`Despite using only three dense FFN layers` 指的是 first_k_dense_replace = 3


**我们将这一过程优化为：在前一个注意力层之后进行一次 reduce-scatter，在下一个层之前进行一次 all-gather，从而将通信成本减少了 50%。**怎么优化的？为什么会减少一半？

### Sparse FFNs

**此外，当注意力机制也在纯 DP 下计算时，设备间的通信将被完全消除，显著提升整体效率。 **为什么？

> 在 DeepSeek-V3 的专家混合（Mixture of Experts，MoE）架构中，稀疏前馈网络（FFN）需要大量的专家权重，从而造成了显著的内存瓶颈。为了解决这一问题，我们采用了专家并行（Expert Parallelism，EP）策略，将专家权重分布在多个设备上。这种方法在保持高性能的同时，有效地扩展了内存容量，但也带来了如不规则的全对全通信和负载不均等挑战。

> 右上图展示了我们基于 DeepEP 框架实现的 EP 方案，关于我们 EP 设计和优化的更多细节将在接下来的章节中介绍。

### LM Head

> 语言模型头（LM head）负责在一个庞大的词汇表上计算输出概率，这是一项资源密集型操作。传统上通过词汇表并行（vocabulary parallelism）来聚合来自张量并行（TP）组的 token logits。为了提升可扩展性和效率，我们采用了数据并行（Data Parallelism，DP）策略，与我们在稠密前馈网络（dense FFN）中采用的方式一致。这样不仅降低了内存开销，也简化了设备间的通信，实现了更加高效简洁的解决方案。


**词汇表并行（vocabulary parallelism）** 具体是怎么切分的？

## Prefill and Decode Disaggregation
