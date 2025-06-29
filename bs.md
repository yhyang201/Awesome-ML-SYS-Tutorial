### SGLang 中的 `batch_size` —— 精确定义与各阶段含义

在推理场景里，**`batch_size` 默认指 *token\_batch\_size***：一次 forward 中真正进入 Self-Attention 与 FFN 的 **token 总量**。以下分析均基于此定义。

---

#### Prefill 阶段：

* **场景**
  同时到来的 $n$ 条请求，其上下文长度分别为 $L_0,L_1,\dots,L_{n-1}$。
* **原理**
  Prefill 需要把整段上下文一次性喂入模型，为每个 token 生成 **Key/Value** 并写入 KV-Cache。
  SGLang 采用 *token-level batching*：只要一个 token 落到算子里，就算作本轮有效负载。于是
  
$$
bs = \sum_{i=0}^{n-1} L_i
$$

  

---

#### Decode 阶段：

* **场景**
  解码时，每条活跃序列仅追加 **1 个 query token**（若采用 multi-step 或 beam search，一次追加 $k$ 个）。
* **原理**
  历史 token 已缓存在 KV-Cache 中，无需重算；真正进入 Attention / FFN 的只有这些 **新增 token**。若当前活跃序列数为 $n$，则
  
$$
bs = n
$$


(在 Attention 时，会复用KV Cache，因此每个序列 Decode 的 Query Token 长度为 1，KV Token 长度为当前序列长度。)

### 进阶主题

#### DP Attention 与 `batch_size`

| 模块            | 并行策略                               | `token_batch_size` 计算       |
| ------------- | ---------------------------------- | --------------------------- |
| **Attention** | Data Parallel (DP)                 | 各 DP-rank 独立                |
| **FFN**       | Tensor / Expert Parallel (TP / EP) | 为 *每个* DP-rank 的本地 batch 之和 |

---

#### CUDA Graph 与 `batch_size`

* `cuda_graph_runner` 会预先 **capture** 一组离散 batch 尺寸 `1, 2, 4, … 256`。
* 实际执行时，若当前 `BS` 不在列表中，将取**第一个 ≥ BS** 的捕获值 `cudagraph_bs`，并用若干 `seq_len = 1` 的填充序列补齐 `cudagraph_bs - BS`。
* 若 `BS` 超过列表最大值，则本轮 Decode 不使用 Cuda Graph ，使用正常的 Decode Forward。

> 源码：`sglang/srt/model_executor/cuda_graph_runner.py#L627`

---

#### DeepEP & `num_max_dispatch_tokens_per_rank`

DeepEP 的 *low-latency kernels* 受限于 `num_max_dispatch_tokens_per_rank`：

> “We recommend that the actual batch size in the decoding engine be **< 256**.”

在 sglang 中，可通过环境变量

```bash
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
```

进行配置。
参考仓库：[https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)

