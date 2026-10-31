---
id: 11.15
title: RWKV 与线性注意力：把 softmax 换成可累加的状态
module: M11
slug: rwkv-linear-attention
prereq: [11.13]
minutes: 45
level: 进阶
tags: [RWKV, 线性注意力, RetNet, GLA, DeltaNet]
llm_hooks: [architecture, linear-attention, rnn, long-context]
---

# 11.15 RWKV 与线性注意力：把 softmax 换成可累加的状态

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[11.13 架构全景](13-架构全景.md) ｜ **预计**：45 分钟
>
> **一句话**：线性 attention 把 $\operatorname{softmax}(QK^\top)V$ 改写成“先累加 $K^\top V$，再用 $Q$ 读”，把 $T^2$ 变成 $T$；RWKV 则把同一件事写成一个带衰减的 RNN。

## 0. 场景引入

你要在手机上跑一个 128k 的本地助手。全量 attention 的 KV 太大，Mamba 的 scan kernel 又不一定在手机 GPU 上高效。另一条路是**线性 attention/RWKV**：

- 训练时像 Transformer 一样并行；
- 推理时像 RNN 一样逐 token，状态固定；
- 不要 KV Cache，不要 $T^2$ 分数矩阵。

但它也有代价：把 softmax 的“精确内容寻址”换成了“可累加的核特征”，回看能力下降。

**问题**：线性 attention 到底怎么把 $T^2$ 变 $T$？RWKV 的 WKV 递推和它是什么关系？为什么这类模型在长流式/边缘场景有吸引力，却很少单独统治通用大模型？

## 1. 直觉

标准 attention：

```text
每个 query 和所有 key 打分 -> softmax -> 加权求和所有 value
```

线性 attention：

```text
先把所有 (key, value) 累加成一个状态 S
每个 query 直接用 S 读
```

类比：

- 标准 attention 是“每次提问都翻遍所有档案”；
- 线性 attention 是“边看边做索引/摘要，提问时查摘要”。

RWKV 更像“带遗忘的摘要”：旧状态按指数衰减，新信息按门控写入；既能并行训练，也能逐 token 推理。

> 💡 **一句话直觉**：**线性 attention 用“可累加状态”换掉“两两打分”；RWKV 给这个状态加上遗忘门。**

这个类比在哪里失效：摘要会丢细节；softmax 的归一化/尖锐选择被换成了核函数，表达能力不同；线性 attention 的 $O(TD^2)$ 在短序列不一定比 $O(T^2D)$ 便宜。

## 2. 定义

### 2.1 [线性 attention](#ref-katharopoulos-2020)

标准 attention：

$$
O_t=\frac{\sum_{s\le t}\exp(q_t^\top k_s)v_s}{\sum_{s\le t}\exp(q_t^\top k_s)}
$$

把 $\exp(q^\top k)$ 换成核特征 $\phi(q)^\top\phi(k)$：

$$
O_t=\frac{\phi(q_t)^\top\sum_{s\le t}\phi(k_s)v_s^\top}{\phi(q_t)^\top\sum_{s\le t}\phi(k_s)}
$$

定义状态：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad z_t=z_{t-1}+\phi(k_t)
$$

则：

$$
O_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
$$

- $S_t\in\mathbb{R}^{d_k\times d_v}$：固定大小状态；
- 计算 $O(Td_kd_v)$；
- 没有 $T\times T$ 分数矩阵。

### 2.2 因果线性 attention 的并行形式

训练时可以并行：

$$
S_t=\sum_{s\le t}\phi(k_s)v_s^\top
$$

用前缀和/分块矩阵乘算；推理时逐 token 更新 $S_t$。这就是“训练像 Transformer，推理像 RNN”。

### 2.3 [RWKV](#ref-peng-2023) 的 WKV

RWKV 的 time-mix 是一个带指数衰减的加权 key-value 递推（WKV）：

$$
S_t=a_t\odot S_{t-1}+b_t\otimes v_t,\qquad
o_t=q_t^\top S_t
$$

- $a_t$：衰减/保留门（与位置/通道有关）；
- $b_t$：写入门（由 key 得到）；
- $q_t$：读出；
- $S_t$ 固定大小。

这里省略了 WKV 的归一化分母和数值稳定项；不同 RWKV 版本的具体实现不同，但核心都是“带衰减的线性 attention 递推”。

不同版本的 RWKV 在 $a_t,b_t$ 的参数化、token-shift、数值稳定上不同；**核心都是“线性 attention 的递归形式 + 门控”**。

### 2.4 相关家族

| 模型/方法 | 状态更新 | 特点 |
| --- | --- | --- |
| 线性 attention | $S_t=S_{t-1}+\phi(k_t)v_t^\top$ | 最简单，核函数选择关键 |
| [RetNet](#ref-sun-2023) | 带衰减的 $S_t$ | 训练并行、推理递归 |
| [GLA](#ref-yang-2023) | 门控线性 attention | 数据相关的遗忘 |
| [DeltaNet](#ref-schlag-2021) | $S_t=S_{t-1}(I-\beta k k^\top)+\beta k v^\top$ | 用 delta 规则写入/修正 |
| RWKV | 指数衰减 WKV + token shift | RNN 风格，边缘友好 |

## 3. 为什么这样定义

因为标准 attention 的两个成本都来自“两两比较”：

- 计算：$T^2$ 个分数；
- 显存：$T$ 个 K/V（每层）。

线性 attention 把“比较”换成“累加”：

- 训练：前缀和/分块矩阵乘，仍然并行；
- 推理：一个固定状态，逐 token 更新；
- 不需要 KV Cache。

代价：

- 核函数 $\phi$ 的表达力有限；
- 没有 softmax 的尖锐选择；
- 精确检索/复制能力弱；
- 状态容量固定。

RWKV 在此基础上加入**遗忘/门控**，让模型能“丢掉旧信息”，更适合长流。

## 4. 性质与推导

### 4.1 复杂度

| 方法 | 训练计算 | 推理状态 | 训练并行 |
| --- | --- | --- | --- |
| 标准 attention | $O(T^2D)$ | KV $O(TD)$ | 是 |
| 线性 attention | $O(TD^2)$ | $O(D^2)$ | 是（前缀和） |
| RWKV | $O(TD^2)$ | $O(D^2)$ | 是（扫描/分块） |

当 $T\gg D$ 时，线性 attention 更省；当 $T$ 小、$D$ 大时，$TD^2$ 可能更大。

### 4.2 为什么状态是 $D\times D$

$S_t=\sum \phi(k_s)v_s^\top$：$\phi(k_s)\in\mathbb{R}^{d_k}$，$v_s\in\mathbb{R}^{d_v}$，所以 $S_t$ 是 $d_k\times d_v$。多头时每个头一个状态。

### 4.3 数值稳定

线性 attention 的累加可能随 $T$ 增大；RWKV 用指数衰减/最大值技巧稳定 WKV。没有 softmax 的归一化，训练时要注意 scale/初始化。

### 4.4 表达力边界

线性 attention 的核是“可分解”的：$\phi(q)^\top\phi(k)$，不能表示任意 $q,k$ 的交互。这限制了精确匹配/检索。混合少量全量 attention 可以补。

## 5. 例子

### 5.1 128k 的账

标准 attention 7B、32 层、$H_{kv}=32,d_k=128$：128k KV 约 64 GiB。  
线性 attention/RWKV：每层状态 $d_k\times d_v$（例如 $128\times128$），32 层约几 MiB，与 $T$ 无关。

### 5.2 RWKV 的 token shift

RWKV 在 time-mix/channel-mix 前做 token shift：

$$
\tilde x_t=(1-\mu)x_t+\mu x_{t-1}
$$

让当前 token 混合前一个 token 的信息，补局部模式；这是它用很小的成本获得“短程卷积”效果的方式。

### 5.3 为什么 RWKV 适合边缘

- 推理是 RNN，状态固定，内存可预测；
- 不需要 KV Cache；
- 逐 token 计算，适合流式/低功耗；
- 训练可以并行，不用 BPTT 展开整段。

### 5.4 什么时候不该用

- 需要强精确检索/复制；
- 需要极强的内容寻址（长文档问答的精确引用）；
- 工具链/量化支持不成熟。

这时用 attention 或混合架构。

### 5.5 场景闭环：128k 手机助手

**输入**：8 GB 手机，128k 上下文，离线摘要；不能装 64 GiB KV。

**计算**：7B 全 attention 128k KV 约 64 GiB；线性/RWKV 状态约几 MiB。

**输出**：小模型 + 线性/RWKV/混合 + 4-bit；精确检索用本地检索/短上下文重读。

**成本**：内存/功耗可控；质量/工具链不如 attention 成熟。

**失败条件**：任务需要精确引用；量化后质量掉点；kernel 不支持目标手机 GPU。

## 6. 常见误区

1. **“线性 attention 是 attention 的近似”**  
   它是另一种核化注意力，不是 softmax 的数值近似；表达力/行为不同。

2. **“没有 KV 就没有内存”**  
   有固定状态 $S$，只是不随 $T$ 增长。

3. **“RWKV 就是 RNN，不能并行训练”**  
   RWKV 的 WKV 可以写成扫描/分块形式，训练可并行。

4. **“线性 attention 一定更快”**  
   短序列/大 $D$ 时，$TD^2$ 可能大于 $T^2D$；kernel 效率也重要。

5. **“RWKV 能完全替代 Transformer”**  
   精确检索弱；实际多为混合/特定场景。

## 7. 概念网络

- 前置：[11.13 架构全景](13-架构全景.md) —— mixer 分类。
- 前置：[11.4 Attention](04-Attention-下-softmax与多头.md) —— softmax attention 的基线。
- 后续：[11.16 混合架构](16-混合架构.md) —— 线性/RNN + attention。
- 易混：[11.14 Mamba](14-Mamba与SSM.md) —— SSM 是状态空间，线性 attention 是核化 attention；都在做“固定状态”。

## 8. 通往大模型

**出现位置**：RWKV/RetNet/GLA/DeltaNet 模型、边缘推理、长流式应用；Qwen3.8-Flash-Next 的 GatedDeltaNet 属于这一类线性/递归 mixer。

**形状**：$Q,K,V\in\mathbb{R}^{B\times H\times T\times d}$；状态 $S\in\mathbb{R}^{B\times H\times d_k\times d_v}$。

**为什么需要**：不懂线性 attention，你会把“没有 KV”当成“没有状态”，把“线性”当成“一定快”；其实是用状态容量换成本。

**可检验结论**：128k 上下文，7B 全 attention 的 KV 约 64 GiB；线性 attention 的状态只有几 MiB。差距来自 $O(T)$ vs $O(1)$。

一个 4 行的线性 attention 状态更新：

```python
def la_step(S, z, q, k, v, phi):
    S = S + phi(k)[:, None] * v[None, :]
    z = z + phi(k)
    return S, z, (phi(q) @ S) / (phi(q) @ z)
```

## 9. 练习

1. 写出线性 attention 的状态 $S_t$ 与输出 $O_t$；为什么没有 $T\times T$ 矩阵？
2. 线性 attention 的计算复杂度是多少？什么时候不一定比标准 attention 快？
3. RWKV 的 WKV 与线性 attention 的关系是什么？
4. 为什么线性 attention/RWKV 的精确检索弱？怎么补？
5. 128k、7B 全 attention 的 KV 约多少？线性 attention 的状态为什么与 $T$ 无关？

<details>
<summary>参考答案</summary>

1. $S_t=S_{t-1}+\phi(k_t)v_t^\top$，$O_t=\phi(q_t)^\top S_t/(\phi(q_t)^\top z_t)$；所有历史先累加，不需要两两打分。
2. $O(TD^2)$；短 $T$/大 $D$ 时可能大于 $O(T^2D)$。
3. RWKV 的 WKV 是带指数衰减/门控的线性 attention 递推；本质相同，参数化不同。
4. 核化状态是有损摘要；混合少量全量 attention 或检索模块。
5. 约 64 GiB；状态是 $d_k\times d_v$ 的固定矩阵，不随 $T$ 增长。

</details>

## 10. 一页速记

- 线性 attention：$S_t=S_{t-1}+\phi(k_t)v_t^\top$，$O_t=\phi(q_t)^\top S_t/(\phi(q_t)^\top z_t)$。
- 计算 $O(TD^2)$，状态 $O(D^2)$，无 KV；训练可并行，推理像 RNN。
- RWKV = 带衰减/门控的 WKV + token shift；边缘/流式友好。
- 弱点是精确检索/复制；混合 attention 可补。
- “线性”不等于“短序列更快”；看 $T$ 与 $D$ 的比例。

## 11. 参考文献与致谢

- <a id="ref-peng-2023"></a>**Peng 等（2023）**：*RWKV: Reinventing RNNs for the Transformer Era*。RWKV 的 WKV 与 token shift。[arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
- <a id="ref-katharopoulos-2020"></a>**Katharopoulos 等（2020）**：*Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*。线性 attention 的核函数形式。[arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- <a id="ref-sun-2023"></a>**Sun 等（2023）**：*Retentive Network: A Successor to Transformer for Large Language Models*。带衰减的线性 attention/RetNet。[arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
- <a id="ref-yang-2023"></a>**Yang 等（2023）**：*Gated Linear Attention Transformers with Hardware-Efficient Training*。GLA 的门控线性 attention。[arXiv:2312.06635](https://arxiv.org/abs/2312.06635)
- <a id="ref-schlag-2021"></a>**Schlag 等（2021）**：*Linear Transformers with Learnable Adaption*。DeltaNet 的 delta 写入规则。[arXiv:2102.11174](https://arxiv.org/abs/2102.11174)

---

上一篇 → [11.14 Mamba 与 SSM](14-Mamba与SSM.md) ｜ 下一篇 → [11.16 混合架构](16-混合架构.md)
