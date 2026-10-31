---
id: 11.14
title: Mamba 与选择性状态空间：把历史压进一个状态
module: M11
slug: mamba-ssm
prereq: [11.13]
minutes: 45
level: 进阶
tags: [Mamba, SSM, 状态空间, 选择性, 长上下文]
llm_hooks: [architecture, ssm, long-context, inference]
---

# 11.14 Mamba 与选择性状态空间：把历史压进一个状态

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[11.13 架构全景](13-架构全景.md) ｜ **预计**：45 分钟
>
> **一句话**：Mamba 用“输入相关的状态转移”代替 attention 的显式 KV：每层只维护一个固定大小的状态，逐 token 更新，计算 $O(T)$、内存 $O(1)$。

## 0. 场景引入

你要做一个 256k 的流式日志助手：输入持续到来，模型要一边读一边总结，不能等整段文本结束。

标准 attention 的做法是缓存所有历史 K/V：256k 时每层 KV 约 128 GiB（7B 全 attention 估算），而且每来一个 token 都要读一遍。Mamba 的做法是：每层维护一个 $D\times N$ 的状态（$N$ 通常 16），逐 token 更新；不管上下文多长，状态大小不变。

代价是：状态是“有损摘要”。问“第 137 页第 3 行是什么”，它可能答不出来；问“这段日志的趋势是什么”，它很擅长。

**问题**：状态空间模型到底在算什么？为什么“选择性”是 Mamba 的关键？它和 attention 各自适合什么？

## 1. 直觉

把 SSM 想成一个带“记忆旋钮”的录音机：

- 每个时刻输入 $x_t$；
- 内部状态 $h_t$ 是“当前还记得的东西”；
- 输出 $y_t$ 是从状态里读出的结果；
- 状态转移 $A$ 决定“旧记忆保留多少”；
- 输入矩阵 $B$ 决定“新信息写多少”；
- 输出矩阵 $C$ 决定“读哪部分”。

标准 SSM 的 $A,B,C$ 与输入无关：不管看到什么，都按同一规则记/忘。Mamba 的“选择性”让 $B,C,\Delta$ 依赖当前输入：**看到重要 token 就多写，看到无关 token 就少写**。

> 💡 **一句话直觉**：**attention 是“把所有历史摊在桌上”，SSM 是“把历史压进一个笔记本”；Mamba 让笔记本的写法随内容变化。**

这个类比在哪里失效：笔记本容量固定，写满就丢；attention 的“桌子”随 $T$ 变大，但读取也变贵；笔记本的“读”是线性的，不是内容寻址。

## 2. 定义

### 2.1 连续时间 SSM

$$
h'(t)=A h(t)+B x(t),\qquad y(t)=C h(t)+D x(t)
$$

- $h(t)\in\mathbb{R}^N$：状态；
- $A\in\mathbb{R}^{N\times N}$：状态转移；
- $B\in\mathbb{R}^{N\times1}$、$C\in\mathbb{R}^{1\times N}$：写入/读出；
- $D$：跳连（skip）。

### 2.2 离散化

用步长 $\Delta$ 把连续系统变成递推：

$$
h_t=\bar A h_{t-1}+\bar B x_t,\qquad y_t=C h_t+D x_t
$$

一种常见离散化（零阶保持）：

$$
\bar A=\exp(\Delta A),\qquad
\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\Delta B
$$

S4 一类早期 SSM 见 [S4](#ref-gu-2021)。Mamba 的实现常写成更简单的形式：

$$
h_t=\exp(\Delta_t A)h_{t-1}+\Delta_t B_t x_t
$$

其中 $A$ 取对角负值，保证稳定。

### 2.3 选择性（selection）

Mamba 让 $\Delta_t,B_t,C_t$ 由输入 $x_t$ 线性投影得到：

$$
\Delta_t=\operatorname{softplus}(W_\Delta x_t+b_\Delta),\qquad
B_t=W_B x_t,\qquad C_t=W_C x_t
$$

- $\Delta_t$ 大：多写新信息、少保留旧状态；
- $\Delta_t$ 小：多保留旧状态、少写新信息。

这就是“选择性”：模型可以学会“这个 token 重要，记下来；那个 token 不重要，跳过”。

### 2.4 [Mamba](#ref-gu-2023) block

一层 Mamba 通常是：

```text
x -> in_proj -> conv1d -> SiLU -> SSM(selective scan) -> * gate -> out_proj
```

- `in_proj`：把 $D$ 投到 $2D$（一路 SSM，一路 gate）；
- `conv1d`：局部卷积，补短程模式；
- `SSM`：核心选择性扫描；
- `gate`：SiLU/门控，类似 GLU；
- `out_proj`：回到 $D$。

### 2.5 [Mamba-2](#ref-dao-2024) 与 SSD

Mamba-2 把状态空间写成“结构化状态空间对偶”（SSD），用矩阵乘/张量核加速，支持多头；$A$ 简化为标量乘单位阵，训练更快。它仍然是“固定状态 + 线性时间”，不是 attention。

## 3. 为什么这样定义

因为标准 SSM 的 $A,B,C$ 固定，无法根据内容选择“记什么”。语言里，哪个 token 重要高度依赖上下文：

- 看到“但是”要记住转折；
- 看到重复的模板要跳过；
- 看到数字/实体要精确写入。

Mamba 的输入相关 $\Delta,B,C$ 让“写/忘/读”变成可学习的函数。代价是：不再能用一个固定卷积核并行算完整序列，需要**并行扫描（parallel scan）**；GPU 上要做专门 kernel。

## 4. 性质与推导

### 4.1 计算与内存

对长度 $T$、隐藏维 $D$、状态维 $N$：

- 计算：$O(TDN)$；
- 状态内存：$O(DN)$，与 $T$ 无关；
- 没有 KV Cache；
- 训练：并行扫描，$O(T)$ 深度；
- 推理：逐 token 递推，$O(1)$ 状态。

### 4.2 为什么是“线性时间”

展开递推：

$$
h_t=\sum_{k=1}^{t}\left(\prod_{j=k+1}^{t}\bar A_j\right)\bar B_k x_k
$$

每个 $x_k$ 对 $h_t$ 的贡献是一个衰减的核；不需要像 attention 那样两两比较所有 token。因此 $T$ 线性，而不是 $T^2$。

### 4.3 表达力边界

状态只有 $N$ 维/通道，信息被压缩。它擅长：

- 长流式摘要；
- 趋势/模式；
- 局部卷积 + 门控。

它不擅长：

- 精确复制/检索（“第 137 行”）；
- 内容寻址（“找到和这个 query 最像的 token”）；
- 训练时“每个 token 看所有 token”的灵活路由。

这就是为什么实际模型常用**混合**：少量 attention 层补精确回看。

### 4.4 训练与推理的“形状”

| 阶段 | 做法 | 复杂度 |
| --- | --- | --- |
| 训练 | 并行扫描/分块矩阵乘 | $O(TDN)$，可并行 |
| 推理 | 逐 token 递推 | 每步 $O(DN)$，状态常驻 |

推理时不需要重算历史，也不需要存 KV；但每步都要更新状态，GPU 利用率不如大 batch attention 的 GEMM 那么高。

## 5. 例子

### 5.1 1.4B Mamba 的账

假设 $D=2048$、$N=16$、24 层：

- 每层状态约 $D\times N=32{,}768$ 个元素；fp16 约 64 KiB；
- 24 层约 1.5 MiB/序列，与上下文长度无关；
- 7B 全 attention 在 32k 的 KV 约 16 GiB。

**Mamba 的“内存优势”来自结构，不是量化。**

### 5.2 Mamba-2 的 SSD

Mamba-2 把递推写成矩阵乘形式，利用 tensor core；在训练/长序列上更快。它仍然是固定状态，不会变成 attention。

### 5.3 [Jamba](#ref-liebert-2024)：Mamba + attention + MoE

Jamba 交错 Mamba 层、attention 层和 MoE：Mamba 处理长流，attention 补精确回看，MoE 扩容量。这是“纯 SSM 不够、纯 attention 太贵”的工程答案。

### 5.4 什么时候不该用纯 Mamba

- 任务需要逐字复制/检索长文档；
- 需要强内容寻址（RAG 的精确引用）；
- 训练/推理栈只支持 attention（工具链不成熟）。

这时用混合架构，或直接用 attention。

### 5.5 场景闭环：1M 流式日志

**输入**：1M token 日志，持续输入，单卡；目标是趋势/异常摘要。

**计算**：Mamba 状态约 1.5 MiB/序列，与 $T$ 无关；7B 全 attention 1M KV 约 512 GiB。

**输出**：Mamba/混合做流式摘要；需要精确引用某行时，加检索或少量 attention。

**成本**：状态常驻显存很小；训练/工具链较新。

**失败条件**：任务要求逐字检索；scan kernel 在目标硬件效率低；状态容量不够。

## 6. 常见误区

1. **“Mamba 是 attention 的完全替代”**  
   它擅长长流式，不擅长精确检索；实际多为混合。

2. **“SSM 没有状态”**  
   它有固定大小的状态，只是不随 $T$ 增长。

3. **“Mamba 一定比 Transformer 快”**  
   短序列/大 batch 下，attention 的 GEMM 很成熟；Mamba 的 scan kernel 有开销。

4. **“状态越大越好”**  
   $N$ 增大提升容量，也增加计算/内存；要权衡。

5. **“Mamba-2 是另一种 attention”**  
   它是 SSM 的结构化对偶，仍用固定状态。

## 7. 概念网络

- 前置：[11.13 架构全景](13-架构全景.md) —— mixer 的分类。
- 前置：[11.7 KV Cache](07-KVCache.md) —— attention 的状态为什么随 $T$ 增长。
- 后续：[11.15 RWKV 与线性注意力](15-RWKV与线性注意力.md) —— 另一条线性化路线。
- 后续：[11.16 混合架构](16-混合架构.md) —— Mamba + attention。
- 易混：[11.4 Attention](04-Attention-下-softmax与多头.md) —— attention 是内容寻址，SSM 是状态压缩。

- 服务侧：[13.3 引擎解剖](../13-推理服务/03-引擎解剖.md)、[13.4 调度与并发](../13-推理服务/04-调度与并发.md) —— 固定状态、无 KV、scan kernel。
## 8. 通往大模型

**出现位置**：Mamba/Mamba-2 模型、Jamba 的 Mamba 层、llama.cpp 的 Mamba 支持、长流式推理。

**形状**：输入 `[B,T,D]`；状态 `[B,D,N]`；`Δ/B/C` 由 `x` 投影得到；输出 `[B,T,D]`。

**为什么需要**：不懂 SSM，你会把“没有 KV”当成“没有内存”，把“状态压缩”当成“免费长上下文”；其实它用容量换成本。

**可检验结论**：Mamba 的状态内存与 $T$ 无关；7B 全 attention 的 32k KV 约 16 GiB。两者不是同一个数量级。

一个 5 行的状态更新：

```python
import numpy as np
def ssm_step(h, x, A, B, C, D, dt):
    A_bar = np.exp(dt * A)
    h = A_bar * h + dt * B * x
    return h, C @ h + D * x
```

## 9. 练习

1. 写出连续 SSM 与离散递推；$\Delta$ 的作用是什么？
2. Mamba 的“选择性”体现在哪些参数上？为什么它重要？
3. 计算：$D=2048,N=16$、24 层、fp16，状态多少 MiB/序列？
4. 为什么 Mamba 不擅长“逐字检索”？混合架构怎么补？
5. Mamba-2 的 SSD 主要解决什么问题？

<details>
<summary>参考答案</summary>

1. $h'=Ah+Bx,y=Ch+Dx$；$h_t=\exp(\Delta A)h_{t-1}+\Delta B x_t$；$\Delta$ 控制写/忘的步长。
2. $\Delta,B,C$ 由输入投影；让模型按内容决定记/忘/读。
3. $2048\times16\times24\times2$ B ≈ 1.5 MiB/序列。
4. 状态是固定容量摘要，不能无损保留每个 token；混合少量 attention 层做精确回看。
5. 用矩阵乘/张量核加速训练与长序列，同时保持固定状态。

</details>

## 10. 一页速记

- SSM：$h_t=\bar A h_{t-1}+\bar B x_t$，$y_t=C h_t+D x_t$；状态固定。
- Mamba 的选择性：$\Delta,B,C$ 由输入决定；$\Delta$ 控制写/忘。
- 计算 $O(TDN)$，状态 $O(DN)$，无 KV；擅长长流式，弱在精确检索。
- Mamba-2/SSD 用矩阵乘/张量核加速；Jamba 用 Mamba+attention+MoE 补短板。
- 长上下文不是“免费”：用状态容量换显存/计算。

## 11. 参考文献与致谢

- <a id="ref-gu-2023"></a>**Gu 与 Dao（2023）**：*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*。选择性 SSM 与并行扫描。[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- <a id="ref-dao-2024"></a>**Dao 与 Gu（2024）**：*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*。Mamba-2/SSD 的结构化对偶。[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- <a id="ref-gu-2021"></a>**Gu 等（2021）**：*Efficiently Modeling Long Sequences with Structured State Spaces*。S4 的长期记忆结构。[arXiv:2110.13985](https://arxiv.org/abs/2110.13985)
- <a id="ref-liebert-2024"></a>**Lieber 等（2024）**：*Jamba: A Hybrid Transformer-Mamba Language Model*。Mamba + attention + MoE 的混合。[arXiv:2403.19887](https://arxiv.org/abs/2403.19887)

---

上一篇 → [11.13 架构全景](13-架构全景.md) ｜ 下一篇 → [11.15 RWKV 与线性注意力](15-RWKV与线性注意力.md)
