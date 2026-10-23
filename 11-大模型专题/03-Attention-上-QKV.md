---
id: 11.3
title: Attention（上）：QKV 与软查表
module: M11
slug: attention-qkv
prereq: [1.3, 11.2]
minutes: 45
level: 核心
tags: [attention, QKV, 点积, softmax, 因果掩码]
llm_hooks: [self-attention, causal-mask, kv-cache]
---

# 11.3 Attention（上）：QKV 与软查表

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[1.3](../01-向量与几何/03-点积.md)、[11.2](02-线性层.md) ｜ **预计**：45 分钟
>
> **一句话**：attention 让每个 token 用自己的 query 去「问」所有 token 的 key，按匹配度加权取回它们的 value——一次可微的软查表。

## 0. 场景引入

句子：「那只动物没有过马路，因为它太累了。」

「它」指什么？人类一看就知道是「动物」，不是「马路」。模型怎么做到？

在 attention 里，处理「它」这个位置时，会生成一个 query 向量；每个历史 token 生成 key 向量。
query 和「动物」的 key 点积很大，和「马路」的 key 点积较小。softmax 后，「动物」的权重高，于是「它」的新表示主要吸收了「动物」的信息。

**问题**：Q、K、V 分别是什么？为什么用点积？mask 和缩放怎么回事？形状如何变化？

## 1. 直觉

把 attention 想成一次**软查表**：

- **Query（Q）**：我想找什么？（当前 token 的提问）
- **Key（K）**：我有什么标签？（每个 token 的索引）
- **Value（V）**：如果你选中我，我给你什么内容？（每个 token 的内容）

硬查表是「找到完全匹配的那一条，取它的 value」；软查表是「给每条打一个相似度分数，按分数加权平均所有 value」。

> 💡 **一句话直觉**：**attention = 用点积算「问题与标签的匹配度」，softmax 变成权重，再对 value 加权求和。**

类比边界：它不是数据库检索，Q/K/V 都是**学出来的**线性投影，匹配度不等于语义相关性；attention 权重也不等于「解释」。

## 2. 定义

### 2.1 输入与投影

设输入 $X\in\mathbb{R}^{T\times D}$（单个头、单条序列）。定义

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

其中 $W_Q,W_K\in\mathbb{R}^{D\times d_k}$，$W_V\in\mathbb{R}^{D\times d_v}$。常见 $d_k=d_v=D/H$。

### 2.2 缩放点积注意力

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

逐项读法：

- $QK^\top\in\mathbb{R}^{T\times T}$：第 $(i,j)$ 项是第 $i$ 个 query 与第 $j$ 个 key 的点积；
- 除以 $\sqrt{d_k}$：控制方差（第 8 节）；
- softmax 按**行**归一化：每个 query 对所有 key 的权重和为 1；
- 乘以 $V$：得到每个位置的新表示，形状 $[T,d_v]$。

### 2.3 因果掩码

自回归生成时，第 $i$ 个位置不能看到 $j>i$。做法是把上三角的分数设为 $-\infty$：

$$
\tilde{S}_{ij}=\begin{cases}
S_{ij}, & j\le i\\
-\infty, & j>i
\end{cases}
$$

softmax 后这些位置的权重为 0。

### 2.4 多头

$$
\operatorname{MultiHead}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_H)W_O
$$

每个头有独立的 $W_Q^{(h)},W_K^{(h)},W_V^{(h)}$，在 $d_k=D/H$ 维子空间里做 attention，最后拼接再投影（第 11.4 节）。

## 3. 为什么这样定义

**为什么用点积而不是欧氏距离？** 点积是双线性的，计算可以用一次矩阵乘法完成，GPU 友好；而且它衡量「方向一致 + 强度」，适合做匹配分数。

**为什么要 Q 和 K 分开？** 同一个 token 作为「提问者」和「被查询者」的角色不同。分开投影让模型可以学「我需要什么」与「我能提供什么」两套表示。

**为什么要 V？** K 只负责匹配，V 负责传递内容。若直接用 K 加权，匹配与内容耦合，表达力受限。

**为什么 softmax？** 它把任意实数分数变成非负、和为 1 的权重，且可微；同时具有「赢者通吃」的趋势（分数差越大越集中）。

## 4. 性质与推导

### 4.1 形状链

$$
X:[T,D]\ \to\ Q,K:[T,d_k],\ V:[T,d_v]
$$
$$
QK^\top:[T,T]\ \to\ \text{softmax}:[T,T]\ \to\ AV:[T,d_v]
$$

批量与多头：

$$
X:[B,T,D]\to Q,K,V:[B,H,T,d_k]\to S:[B,H,T,T]\to O:[B,H,T,d_k]\to[B,T,D]
$$

### 4.2 每行权重和为 1

softmax 的定义保证对每个 $i$：$\sum_j A_{ij}=1$。因此输出是 value 的**凸组合**，不会因序列变长而数值爆炸（但权重可能集中在少数位置）。

### 4.3 计算复杂度

$QK^\top$ 需要 $O(T^2d_k)$，$AV$ 需要 $O(T^2d_v)$，合计 $O(T^2d)$。这就是长上下文昂贵的根源；FlashAttention 通过分块减少显存，但不改变 $O(T^2)$ 的计算量。

### 4.4 排列等变性（无位置编码时）

若把输入 token 的顺序打乱，输出也会同样打乱——attention 本身不知道顺序。所以必须加位置编码（第 11.5 节）。

### 4.5 分数矩阵的秩

$QK^\top$ 的秩至多 $\min(T,d_k)$。当 $d_k\ll T$ 时，分数矩阵是低秩的；softmax 是非线性的，会改变秩，但注意力仍受这个结构约束。

## 5. 例子

### 5.1 手算：两个 token 的 self-attention

设 $X=\begin{bmatrix}1&0\\0&1\end{bmatrix}$（两个 token，$D=2$），$W_Q=W_K=W_V=I$，$d_k=2$。

$$
Q=K=V=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

$$
QK^\top=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
\frac{QK^\top}{\sqrt2}=\begin{bmatrix}0.707&0\\0&0.707\end{bmatrix}
$$

对每行做 softmax（$e^{0.707}\approx2.028$，$e^0=1$）：

$$
A=\begin{bmatrix}0.670&0.330\\0.330&0.670\end{bmatrix}
$$

$$
AV=\begin{bmatrix}0.670&0.330\\0.330&0.670\end{bmatrix}
$$

结果：每个 token 的新表示是「自己 67% + 对方 33%」。因为两个 token 的 query/key 正交，匹配度相同，softmax 把分数差（0.707 vs 0）压成了 2:1 的权重。

**若加因果掩码**：第 1 行只能看自己，$A=\begin{bmatrix}1&0\\0.330&0.670\end{bmatrix}$，输出第 1 个 token 保持 $(1,0)$。

### 5.2 手算：为什么除以 $\sqrt{d_k}$

设 $q,k$ 各分量独立 $\mathcal{N}(0,1)$，则

$$
q^\top k=\sum_{i=1}^{d_k}q_ik_i
$$

每项方差 1，独立相加，方差为 $d_k$，标准差 $\sqrt{d_k}$。
$d_k=128$ 时标准差约 11.3；softmax 输入跨度过大，输出接近 one-hot，梯度接近 0。除以 $\sqrt{d_k}$ 后方差为 1。

### 5.3 场景：一个注意力头的完整形状

```python
import numpy as np
B, T, D, H = 2, 5, 8, 2
d_k = D // H
X = np.random.randn(B, T, D)
Wq = np.random.randn(D, D)
Q = (X @ Wq).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # [B,H,T,d_k]
S = Q @ Q.transpose(0, 1, 3, 2) / np.sqrt(d_k)             # [B,H,T,T]
A = np.exp(S - S.max(axis=-1, keepdims=True))
A = A / A.sum(axis=-1, keepdims=True)
print(A.shape, A.sum(axis=-1)[0, 0])  # (2,2,5,5) [1. 1. 1. 1. 1.]
```

## 6. 常见误区

1. **误区**：attention 权重就是「模型解释」。
   **纠正**：权重只表示信息混合比例，不表示因果重要性；不同头、不同层可能相互补偿。

2. **误区**：Q、K、V 是输入的不同副本。
   **纠正**：它们是输入经过**不同**可学习投影得到的；初始化相同时也会因梯度不同而分化。

3. **误区**：attention 是「记忆」。
   **纠正**：它只访问当前上下文窗口内的 token；长期知识存储在权重里，不在 attention 里。

4. **误区**：mask 只是「把未来置零」。
   **纠正**：是在 softmax 前置 $-\infty$，保证归一化后严格为 0；若在 softmax 后置零，行和不再为 1。

## 7. 概念网络

- 前置：[1.3 点积](../01-向量与几何/03-点积.md) —— 匹配分数。
- 前置：[11.2 线性层](02-线性层.md) —— QKV 投影。
- 后续：[11.4 多头与 softmax](04-Attention-下-softmax与多头.md) —— 并行子空间。
- 后续：[11.7 KV Cache](07-KVCache.md) —— 缓存 K、V。
- 关联：[11.5 RoPE](05-位置编码与RoPE.md) —— 给 Q/K 注入位置。

## 8. 通往大模型：参数量、复杂度与实现

**参数量**（$W_Q,W_K,W_V,W_O$ 各 $D\times D$）：$4D^2$。
对 $D=4096$：约 6710 万/层。

**计算量**：$O(T^2d)$；$T=4096,d=4096$ 时约 $6.9\times10^{10}$ FLOPs/层（仅 $QK^\top$ 与 $AV$），是长上下文的主要开销。

**KV Cache**：推理时缓存历史 $K,V$，每步只算新 token 的 Q，避免重复计算历史（第 11.7 节）。

**可检验结论**：因果 attention 的权重矩阵是下三角（$A_{ij}=0$ 对 $j>i$），且每行和为 1。若你的实现出现上三角非零，说明 mask 位置或 softmax 顺序错了。

## 9. 练习

1. **概念**：为什么 attention 要分成 Q、K、V 三个投影，而不是直接用输入两两做点积？
2. **计算**：$T=1024,d_k=64$，$QK^\top$ 有多少个元素？需要多少次乘加（数量级）？
3. **计算**：接 5.1，若把 $W_Q$ 换成 $2I$，分数和权重会怎样变化？输出会变吗？
4. **场景**：为什么因果 mask 必须在 softmax **之前**加 $-\infty$？在之后置零会破坏什么？
5. **开放**：如果 attention 权重矩阵的秩很低，对模型表达有什么影响？可能带来什么计算优势？

<details>
<summary>参考答案</summary>

1. 直接用输入点积会让「提问」和「被查询」共用同一表示，无法分别学习；Q/K/V 分离提供更大的表达空间，V 还负责内容传递。
2. $1024^2\approx1.05$M 个元素；每个元素 $d_k=64$ 次乘加，共约 $6.7\times10^7$ 次乘加（$1.34\times10^8$ FLOPs）。
3. 分数翻倍，softmax 更尖锐（对角线权重更高），输出更接近自身；会变。
4. softmax 后再置零会破坏「每行和为 1」，输出不再是凸组合；正确的做法是 softmax 前把非法位置设为 $-\infty$。
5. 秩低意味着所有 query 的注意力模式高度相似，减少头的多样性、降低表达；优势是可低秩近似、加速长序列计算。

</details>

## 10. 一页速记

- $Q=XW_Q$，$K=XW_K$，$V=XW_V$；$A=\operatorname{softmax}(QK^\top/\sqrt{d_k})$；输出 $AV$。
- 形状：$[B,H,T,d_k]\to[B,H,T,T]\to[B,H,T,d_k]$。
- 缩放为了控制点积方差；mask 在 softmax 前加 $-\infty$。
- 复杂度 $O(T^2d)$；因果 attention 权重下三角、行和为 1。
- 参数量约 $4D^2$/层。

---

下一篇 → [11.4 Attention（下）：softmax 与多头](04-Attention-下-softmax与多头.md)
