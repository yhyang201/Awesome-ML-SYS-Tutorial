
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

> 大型语言模型（LLM）的推理包含两个截然不同的阶段：预填充（Prefill）和解码（Decode）。预填充阶段是计算密集型的，需要处理整个输入序列；而解码阶段是内存密集型的，需要管理用于生成token（词元）的键值缓存（KV cache）。传统上，这两个阶段在统一的引擎内处理，其中预填充批次和解码批次的混合调度会引入效率低下的问题。为应对这些挑战，我们在SGLang中引入了预填充与解码（PD）分离的技术。

### Issues with Unified Scheduling

> 传统的统一引擎将 prefill（预填充）和 decode（解码）批次一同处理，导致以下三个主要问题：
> 
> 1. **Prefill 中断**：新的 prefill 批次经常打断正在进行的 decode 批次，从而显著延迟了 token 的生成。
> 2. **数据并行注意力失衡（DP Attention Imbalance）**：在 DP attention 模式下，一个 DP 工作线程可能正在处理 prefill 批次，而另一个则处理 decode 批次，导致解码延迟增加。
> 3. **与 DeepEP 不兼容**：正如我们将在后文中讨论的，DeepEP 在 prefill 和 decode 阶段采用了不同的调度模式，这使得统一调度方式无法与 DeepEP 兼容。
>
> PD 分离（PD Disaggregation）通过将这两个阶段分开处理，解决了上述问题，使每个阶段都能进行针对性的优化。

2了话，在 DP Attention 的情况下，一定会开启 mixed prefill decode 吗？

3了话，所以 DeepEP 必须要 PD 分离吗？

### Implementation Details

> SGLang 中的 PD 分离（PD Disaggregation）设计如下图所示，通过 Prefill 服务器和 Decode 服务器交错执行各自的任务：
>
> ![image](https://github.com/user-attachments/assets/7c8fdbea-2a77-4d8b-95aa-c9982082c868)
>
> 在接收到输入请求后，工作流程如下进行：
>
> 1. Prefill 服务器与 Decode 服务器通过握手配对，分别建立本地的发送端和接收端。
> 2. Decode 服务器预先分配好 KV 缓存，并通知 Prefill 服务器开始模型前向计算，生成 KV 缓存。
> 3. KV 缓存计算完成后，数据被传输至 Decode 服务器，由其执行后续的迭代式 token 生成。
> 
> 这种分离式设计确保每个阶段都在最优条件下运行，从而最大化 GPU 资源的利用率。为了进一步提升性能，我们的实现还包括：
>
> * **非阻塞传输**：数据的发送与接收在后台线程中进行，不会阻塞调度器的事件循环。
> * **基于 RDMA 的传输**：远程直接内存访问（RDMA）通过队列对（queue pairs）建立连接，并使用分散-聚集元素（SGE）高效传输非连续内存块。
> * **灵活的 API 集成**：SGLang 提供灵活的 API，支持集成高性能 RDMA 库，如 Mooncake 和 NIXL，从而简化数据传输过程。
>
> 更多细节请参见我们的设计文档。

## Large-scale Expert Parallelism

### Expert Parallelism with DeepEP

> 由 DeepSeek 团队开发的 DeepEP 是一个通信库，旨在简化 MoE（专家混合）模型中的专家并行（EP）通信。**它解决了在多 GPU 环境下高效地将 token 路由到特定专家的挑战。**通过提供优化的通信内核，DeepEP 能有效降低延迟、提升吞吐量，非常适合用于大规模推理任务。
> 
> DeepEP 提供了两种针对不同工作负载需求的专用调度模式：
> 
> 1. **普通调度（Normal Dispatch）**：优化用于处理较长的输入序列（如 prefill 阶段），该模式优先考虑最大计算吞吐量。但它会生成符号形状（symbolic shapes），这与 CUDA Graph 不兼容，因此在解码（decode）阶段表现较差，因为内核启动的开销会成为显著瓶颈。
> 
> 2. **低延迟调度（Low-Latency Dispatch）**：专为解码阶段生成输出 token 而设计，该模式以最小延迟为优先，确保实时性能。它支持 CUDA Graph，但需要预先分配固定的内存大小。如果实际内存需求超出预> 分配范围，则会导致运行时错误。
> 
> 在 SGLang 中，DeepEP 的集成提供了自动模式（auto mode），可根据工作负载动态选择这两种调度模式。但如果没有 PD 分离（PD Disaggregation），自动模式存在一个限制：在同一个通信组内，不能同时支持普> 通调度（用于 prefill）和低延迟调度（用于 decode）。这一限制阻碍了其与数据并行注意力（DP attention）的兼容性，而 DP attention 对于实现内存高效的推理至关重要。
>
> 下表展示了每种模式的兼容性情况：
>
> 
> | 模式           | 长输入支持 | 长输出支持 | 支持 DP Attention | 支持 CUDA Graph |
> |----------------|------------|-------------|--------------------|------------------|
> | Normal         | ✅         | ❌          | ✅                 | ❌               |
> | Low-Latency    | ❌         | ✅          | ✅                 | ✅               |
> | Auto           | ✅         | ✅          | ❌                 | ✅               |
>
> PD 分离（PD disaggregation）通过将 prefill 和 decode 阶段分开处理，解决了这一问题。它使得在数据并行注意力（DP attention）下，可以对 prefill 阶段使用普通调度，对 decode 阶段使用低延迟调度。通过根据每个阶段的具体需求选择合适的调度模式，这种集成方式优化了资源利用率，提升了整体性能。

### DeepGEMM Integration

> DeepGEMM 是由 DeepSeek 团队开发的另一个高效计算库，专门用于优化 MoE（专家混合）模型中的计算。它提供了两种专用的矩阵乘法函数（Grouped GEMMs），分别适用于推理过程中的不同阶段。
> 
> * **Grouped GEMMs（连续布局）**：该内核适用于动态输入形状，非常适合 MoE 推理中的 prefill 阶段。它处理的数据将来自不同专家的输入按连续方式拼接，从而灵活应对不同输入大小的需求。
>
> * **Grouped GEMMs（掩码布局）**：该内核假设固定的输入形状，并使用掩码张量（mask tensor）只对输入中的有效部分进行计算。它兼容 CUDA Graph，可优化内核启动开销，因此非常适用于对延迟要求极高的 decode 阶段。
> 
> DeepGEMM 与 DeepEP 的调度模式高度兼容：
> 
> * 对于 **连续布局内核**（在 prefill 阶段与 normal dispatch 一起使用），由于 normal dispatch 输出的是符号形状（symbolic shape），需要额外的排列操作，将其转换为内核所需的连续格式。为此，我们参考了 LightLLM 项目，并实现了一个自定义的 Triton 内核来高效完成这一排列过程。该内核确保 normal dispatch 的输出被正确地重新排列，从而可顺利接入连续布局 GEMM 内核。
> 
> * **掩码布局内核** 可无缝配合 DeepEP 的低延迟调度模式，两者都为 decode 阶段优化，并支持 CUDA Graph。
> 
> 此外，SGLang 也将 DeepGEMM 集成用于 MoE 计算，并支持张量并行（Tensor Parallelism）。DeepGEMM 还提供了一个高效的通用 GeMM 内核，在非 MoE 操作中同样具备卓越性能。用户可以通过设置环境变量 > `SGL_ENABLE_JIT_DEEPGEMM=1` 来启用该功能，从而进一步提升计算效率。


### Two-batch Overlap

> 在多节点环境中，受限的通信带宽可能显著增加整体延迟。为了解决这一问题，我们借鉴 DeepSeek 的系统设计，实现了 **Two-batch Overlap（TBO，双批次重叠）**。TBO 将一个批次拆分为两个微批次（micro-batch），实现计算与通信的重叠，同时由于实际批大小减半，也降低了峰值内存占用。然而，将 TBO 实际落地会带来一些实现层面的挑战。

### Implementation Challenges

> 尽管 DeepSeek 公布了 TBO 的设计框架，但在实现过程中仍有两个主要难点：
>
> 1. **代码复杂性**：直接实现 TBO 需要对多个微批次进行管理，容易造成逻辑重复，增加代码维护难度，特别是在微批次数量或重叠场景增加时，更容易出错。
>
> 2. **Prefill 阶段的同步问题**：要实现计算与通信的有效重叠，需要处理 DeepEP 中 normal dispatch 阻塞 CPU 的问题。这种阻塞行为可能导致流水线停滞，使 GPU 空闲，削弱了 TBO 所带来的性能收益。
>
### Abstraction for Clean Implementation

> 为了构建更可维护、可复用的代码，我们引入了基于 **操作（operations）和挂起点（yield points）** 的抽象层。该方法允许我们以处理单个微批次的方式编写逻辑，同时通过插入挂起点，让其他微批次获得执行机会。这样不仅避免了代码重复，也减少了变量名加后缀等繁琐操作；对于某些微批次在一层结束时已完成而其他未完成的情况，也能高效处理。
> 
> 此外，该抽象层还支持轻松扩展到其他重叠策略，例如三批次重叠，仅需少量代码变更即可实现。以下是这种方法的一个简洁示例：
>
> ```python
> operations = [
>    self._forward_attn,
>    YieldOperation(),  # 暂停，切换到其他微批次
>    self._forward_dispatch,
>    self._forward_mlp,
>    YieldOperation(),  # 再次暂停
>    self._forward_combine,
> ]
>
> # 处理单个微批次，无需重复代码
> def _forward_attn(self, state):
>    state.hidden_states = self.self_attn(state.hidden_states, ...)
> ```

### Prefill Overlapping Implementation

> 我们优化了 prefill 阶段的启动顺序，以避免 DeepEP 中 dispatch 操作导致的 CPU 阻塞，即使我们使用的是其异步模式。具体来说：
>
> * **Dispatch 操作会阻塞 CPU，直到 GPU 从其他 rank 接收到元数据以便正确分配张量的大小。**
> * **如果实现不当，在此期间计算流会处于空闲状态，因为没有计算任务被提交到 GPU。**
>
> 为了解决这个问题，我们调整了启动顺序，将计算任务优先提交给 GPU，然后再发起可能阻塞 CPU 的通信操作。这样可以确保 GPU 在通信期间仍保持活跃状态。
> 
> 如下图所示，通过合理的启动顺序，**加粗边框所表示的 TBO 调度路径** 避免了由于 CPU 阻塞（例如 normal dispatch）而产生的“气泡”（空闲期）。
>
> ![image](https://github.com/user-attachments/assets/866d7ccd-0aa6-4f89-a7ec-71772542787c)

### Expert Parallelism Load Balancer

> 在 MoE（专家混合）模型中，专家并行（EP）常常会导致 GPU 之间的工作负载分布不均。这种负载不平衡会使系统必须等待最慢的 GPU 完成计算或通信，造成计算资源浪费，并因专家激活状态而增加内存使用。随着 GPU 数量（即 EP 规模）的增加，这一不平衡问题将变得更加严重。
> 
> 为了解决这个问题，DeepSeek 团队开发了 **专家并行负载均衡器（EPLB, Expert Parallelism Load Balancer）**。EPLB 接收专家分布的统计信息作为输入，并计算出一种最优的专家分布方式，以最小化负载不均。用户可以额外分配一些冗余专家（例如额外增加 32 个专家），与原始的 256 个专家组成一个包含 288 个专家的池。借助这个专家池，EPLB 可以策略性地放置或复制专家，例如：将使用频率最高的专家复制多份，或将中等频率的专家与使用频率极低的专家组合在同一张 GPU 上。
> 
> 除了改善负载均衡外，EPLB 还提供了更灵活的并行化设计能力。在原本只有 256 个专家的情况下，并行规模通常只能是 2 的幂（例如 8、16、32）。而使用 EPLB 的 288 个专家池，则可以支持更多样化的配置，例如并行规模为 12 或 72。
> 
> 在下图中，我们通过模拟展示了系统规模扩大及引入 EPLB 算法对负载不均问题的影响。我们通过 **MoE 层中各 GPU 的平均计算时间与最大计算时间的比值**来衡量 GPU 的“负载均衡度”（balancedness），并用每张 GPU 上处理的 token 数来估算其计算时间。可以看到，随着节点数量的增加，利用率下降，而启用 EPLB 后，系统的利用率得到了显著提升。
>
> ![image](https://github.com/user-attachments/assets/6d42a24f-0d42-4054-9fc4-3279c43cc15f)

### EPLB for Real-World Serving

> 为了让 EPLB（专家并行负载均衡器）发挥最佳效果，其输入的专家分布必须尽可能贴近实际的推理负载。为实现这一目标，有两种策略可以提升分布的匹配度：
>
> 1. **增大批次大小**：更大的 batch 能减少专家使用上的随机波动，从而提升负载均衡性。可以通过扩展集群规模，或采用诸如多 token 预测（Multi-Token Prediction, MTP）等技术来实现。
> 
> 2. **周期性重平衡**：定期更新专家的分布安排，利用时间局部性来优化分布。但这需要高效地重新加载专家，因此必须尽量降低专家重加载操作的开销。
>
> 即使使用了 EPLB，一定程度的负载不均仍然是难以避免的，因此进一步的优化仍是未来值得探索的方向。

### Implementation of Rebalancing

> SGLang 将专家重平衡（expert rebalancing）划分为三个阶段，以确保高效执行并尽可能减少系统干扰：
> 
> 1. **系统加载阶段（System Loading Stage）**：可选择将模型权重从磁盘预加载到主内存中，以加快重平衡过程；也可通过内存映射（mmap）直接从磁盘读取，以降低内存占用。
> 
> 2. **重平衡准备阶段（Rebalance Preparation Stage）**：所需权重在后台异步传输到设备内存，利用空闲的 DMA 硬件引擎进行传输，不会打断正在运行的 GPU 操作。
> 
> 3. **重平衡执行阶段（Rebalance Execution Stage）**：通过设备间的数据拷贝（device-to-device copy）来更新模型权重。该步骤还可以通过物理内存重新绑定（physical memory rebinding）等技术进一步优化。
> 
> 这种分阶段的设计确保了重平衡过程既高效又不会打扰正常推理任务，从而在更新过程中保持系统性能稳定。


# Evaluation
