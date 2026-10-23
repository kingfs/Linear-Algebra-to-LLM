---
id: 11.7
title: KV Cache：形状、显存与带宽
module: M11
slug: kv-cache
prereq: [11.3, 11.4]
minutes: 30
level: 核心
tags: [KV Cache, 推理, 显存, GQA, 带宽]
llm_hooks: [decode, prefill, paged-attention]
---

# 11.7 KV Cache：形状、显存与带宽

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[11.3](03-Attention-上-QKV.md)、[11.4](04-Attention-下-softmax与多头.md) ｜ **预计**：30 分钟
>
> **一句话**：生成每个新 token 时，历史 token 的 K、V 不会变；把它们缓存起来，用显存换计算——这就是 KV Cache。

## 0. 场景引入

生成第 1000 个 token 时，需要让它关注前 999 个 token。如果每一步都把整段序列重新算一遍 attention，总计算量是

$$
O(1^2+2^2+\cdots+T^2)=O(T^3)
$$

生成 4096 个 token 会慢到不可接受。

但实际上，前 999 个 token 的 K、V 在之前已经算过，且**不会改变**。只需缓存它们，新 token 只算自己的 Q，和缓存的 K 做点积即可。

**问题**：缓存到底存什么？占多少显存？为什么说 decode 阶段是「带宽受限」？

## 1. 直觉

把 attention 想成查字典：

- 每来一个新词，你写一张新的索引卡（K）和内容卡（V），放进抽屉；
- 下次要查时，只写一张新的问题卡（Q），然后翻遍抽屉里的所有索引卡；
- 不需要把旧卡重写一遍。

> 💡 **一句话直觉**：**KV Cache = 把历史 token 的 K/V 卡片存起来，新 token 只写自己的 Q 卡。**

类比边界：抽屉有容量上限；序列越长，抽屉越满，显存越紧张。缓存的是**每层、每个头**的 K/V，不是最终输出。

## 2. 定义

### 2.1 缓存内容

对第 $l$ 层、第 $h$ 个头，缓存：

$$
K_{\text{cache}}^{(l,h)}\in\mathbb{R}^{T\times d_k},\qquad
V_{\text{cache}}^{(l,h)}\in\mathbb{R}^{T\times d_v}
$$

批量与多头：

$$
K,V\in\mathbb{R}^{B\times H\times T_{\text{cache}}\times d_k}
$$

### 2.2 单步解码

给定新 token 的隐藏状态 $\mathbf{x}_t$：

$$
\mathbf{q}_t=\mathbf{x}_tW_Q,\qquad
\mathbf{k}_t=\mathbf{x}_tW_K,\qquad
\mathbf{v}_t=\mathbf{x}_tW_V
$$

把 $\mathbf{k}_t,\mathbf{v}_t$ 追加到缓存，然后

$$
\mathbf{s}_t=\frac{\mathbf{q}_tK_{\text{cache}}^\top}{\sqrt{d_k}}\in\mathbb{R}^{1\times T},\qquad
\mathbf{o}_t=\operatorname{softmax}(\mathbf{s}_t)V_{\text{cache}}\in\mathbb{R}^{1\times d_v}
$$

### 2.3 显存公式

$$
\#\text{elements}=2\times B\times H\times T\times d_k
$$

（K、V 各一份。）字节数再乘每个元素的字节数。

对 [GQA](#ref-ainslie-2023)（$H_{kv}$ 组 KV）：

$$
\#\text{elements}=2\times B\times H_{kv}\times T\times d_k
$$

### 2.4 Prefill 与 Decode

- **Prefill**：一次性处理整段 prompt，计算所有 token 的 K/V，是**计算密集**的大矩阵乘；
- **Decode**：逐 token 生成，每步只算一个 Q，是**带宽密集**的瘦矩阵乘。

## 3. 为什么这样定义

如果没有缓存，每生成一个 token 都要重算历史 K/V，计算量随 $T^3$ 增长。缓存把「已经算过的」保存下来，把每步 attention 的计算降到 $O(Td_k)$（对单个新 token）。

代价是显存：缓存大小随 $T$ 线性增长，且每层、每个头都要存。于是有了 GQA/MQA（减少 $H_{kv}$）、量化缓存（减少字节数）、滑动窗口/驱逐（减少 $T$）、[PagedAttention](#ref-kwon-2023)（减少碎片）等一系列优化。

## 4. 性质与推导

### 4.1 计算量对比

| 阶段 | 无缓存 | 有缓存 |
|---|---|---|
| 生成 $T$ 个 token 的 attention | $O(T^3d)$ | $O(T^2d)$（prefill）+ $O(T^2d)$（decode 累计） |

更准确地说，decode 每步 $O(Td)$，$T$ 步累计 $O(T^2d)$。

### 4.2 显存估算

对 $B=1,H=32,T=4096,d_k=128$，fp16（2 字节）：

$$
2\times1\times32\times4096\times128\times2
=67{,}108{,}864\ \text{字节}=64\ \text{MiB}
$$

**单层 64 MiB**；32 层约 2 GiB。这还没算模型权重和激活。

### 4.3 GQA 的收益

若 $H_{kv}=8$，缓存变为原来的 $8/32=1/4$：单层 16 MiB，32 层 512 MiB。

### 4.4 带宽瓶颈

Decode 每步要读取整个 KV Cache 才能算 attention。以 2 GiB 缓存、每秒 100 步为例，仅读缓存就需要 $200$ GiB/s 带宽——已接近高端 GPU 的 HBM 带宽上限。因此长上下文推理往往**不是算力不够，而是带宽不够**。

这笔账不是本页的粗略估计：[Pope 等, 2022](#ref-pope-2022) 把推理拆成 prefill 与 decode 两段，分别核算算力、显存与带宽，结论一致——decode 每步只处理一个 token，算术强度低，时间主要花在搬运权重与 KV Cache 上。

### 4.5 缓存与位置

RoPE 旋转后的 K 被缓存；位置信息已经写进 K 的方向，所以缓存的是「带位置的 K」（第 11.5 节）。

## 5. 例子

### 5.1 手算：单层缓存

$B=1,H=32,T=2048,d_k=128$，fp16：

$$
2\times32\times2048\times128\times2
=33{,}554{,}432\ \text{字节}=32\ \text{MiB}
$$

32 层约 1 GiB。

### 5.2 手算：GQA

同样的模型，$H_{kv}=8$：

$$
2\times8\times2048\times128\times2=8{,}388{,}608\ \text{字节}=8\ \text{MiB}
$$

单层，32 层 256 MiB。

### 5.3 场景：为什么 batch 大会 OOM

缓存大小与 $B$ 成正比。$B=16$ 时，上面的 1 GiB 变成 16 GiB，直接挤爆显存。这就是推理服务要限制并发/上下文长度、或用 PagedAttention 管理碎片的原因。

## 6. 常见误区

1. **误区**：KV Cache 缓存的是输出 logits。
   **纠正**：缓存的是每层每头的 K、V；logits 只与最后一步有关。

2. **误区**：KV Cache 也缓存 Q。
   **纠正**：Q 只用于当前新 token，历史 Q 不再需要。

3. **误区**：KV Cache 能减少显存。
   **纠正**：它用显存换计算；显存占用反而增加。

4. **误区**：缓存大小与层数无关。
   **纠正**：每层都有独立的 K/V，总大小要乘层数 $L$。

## 7. 概念网络

- 前置：[11.3 Attention QKV](03-Attention-上-QKV.md) —— K/V 的来源。
- 前置：[11.4 多头/GQA](04-Attention-下-softmax与多头.md) —— 头数影响缓存。
- 后续：[11.9 量化](09-量化.md) —— 缓存也可以量化。
- 关联：[10.3 GEMM](../10-数值线性代数/03-BLAS与GEMM.md) —— prefill 与 decode 的不同瓶颈。

## 8. 通往大模型：优化手段一览

| 手段 | 作用 | 代价 |
|---|---|---|
| MQA/GQA | 减少 $H_{kv}$ | 表达力略降 |
| KV 量化（int8/fp8） | 减少字节数 | 精度损失 |
| 滑动窗口 | 限制 $T$ | 丢失远距离信息 |
| 驱逐/稀疏 | 只保留重要 token | 策略复杂、可能丢信息 |
| PagedAttention | 分页管理，减少碎片 | 实现复杂 |
| Prefix caching | 复用相同前缀的缓存 | 只对重复前缀有效 |

**可检验结论**：缓存大小 $\propto B\cdot H_{kv}\cdot T\cdot d_k\cdot L\cdot\text{bytes}$。把 $T$ 翻倍，显存翻倍；把 $H_{kv}$ 从 32 降到 8，显存降到 1/4。

## 9. 练习

1. **概念**：为什么 decode 阶段是带宽受限，而 prefill 是计算受限？
2. **计算**：$B=2,H=32,T=8192,d_k=128$，fp16，单层 KV Cache 多少 MiB？32 层呢？
3. **计算**：若用 GQA $H_{kv}=8$，重算第 2 题。
4. **场景**：为什么长上下文推理的吞吐往往受显存带宽限制，而不是 FLOPs？
5. **开放**：如果缓存命中率很低（每个请求的 prompt 都不同），prefix caching 还有意义吗？

<details>
<summary>参考答案</summary>

1. Prefill 一次处理 $T$ 个 token，矩阵乘的 $M=T$ 很大，算力利用率高；decode 每步只处理 1 个 token，矩阵「瘦」，主要时间花在读取权重和缓存上。
2. 元素 $2\times2\times32\times8192\times128=134{,}217{,}728$；fp16 字节 $\times2=268{,}435{,}456\approx256$ MiB/层；32 层约 8 GiB。
3. $H_{kv}=8$：元素为原来的 1/4，单层 64 MiB，32 层 2 GiB。
4. 每生成一个 token 都要把整个缓存读一遍，数据量大而计算量小，瓶颈在 HBM 带宽。
5. 意义有限，但仍可复用于系统提示、few-shot 模板、多轮对话的公共前缀；对完全不同的 prompt 无效。

</details>

## 10. 一页速记

- 缓存每层每头历史 K、V，形状 `[B,H,T,d_k]`。
- 显存 $=2BHTd_k\cdot\text{bytes}\cdot L$。
- GQA/MQA 减少 $H_{kv}$，直接省缓存。
- Prefill 计算密集，decode 带宽密集。
- 缓存用显存换计算，长上下文的核心成本之一。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-kwon-2023"></a>**Kwon 等（2023）**：*Efficient Memory Management for Large Language Model Serving with PagedAttention*。PagedAttention 与 vLLM 的原始论文：把 KV Cache 切成固定大小的块、按页表寻址，消除连续分配带来的显存碎片，正是本页 5.3 节描述的问题。[DOI](https://doi.org/10.1145/3600006.3613165)
- <a id="ref-pope-2022"></a>**Pope 等（2022）**：*Efficiently Scaling Transformer Inference*。把自回归推理拆成 prefill 与 decode 两段，核算算力、显存与带宽，结论与本页 4.4 的估算一致：decode 每步只处理一个 token，瓶颈在搬运权重与 KV Cache 的带宽。[arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- <a id="ref-ainslie-2023"></a>**Ainslie 等（2023）**：*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。GQA 的原始论文：让多个 query 头共享一组 KV 头，把 $H_{kv}$ 从 $H$ 降下来，直接按比例缩小本页 2.3 的缓存公式。[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

---

下一篇 → [11.8 LoRA：低秩假设的工程化](08-LoRA.md)
