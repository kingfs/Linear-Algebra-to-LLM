---
id: 11.4
title: Attention（下）：softmax 与多头
module: M11
slug: attention-softmax-multihead
prereq: [11.3]
minutes: 40
level: 核心
tags: [softmax, 多头注意力, MHA, GQA, 温度]
llm_hooks: [multi-head, gqa, flash-attention]
---

# 11.4 Attention（下）：softmax 与多头

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[11.3](03-Attention-上-QKV.md) ｜ **预计**：40 分钟
>
> **一句话**：softmax 把分数变成「和为 1 的注意力预算」，多头让模型在不同子空间里同时关注不同类型的关系。

## 0. 场景引入

同一个句子里，「它」可能需要在不同层面找指代：

- 语法层：主语是谁？
- 语义层：哪个名词短语是动物？
- 位置层：前一个/后一个词是谁？

一个注意力头很难同时做好所有事。[多头注意力](#ref-vaswani-2017) 的想法是：**开 $H$ 个并行的「频道」，每个频道学一种关系，最后拼接。**

**问题**：softmax 具体怎么算？为什么多头比单头大 $H$ 倍参数？MQA/GQA 又改了什么？

## 1. 直觉

### 1.1 softmax 是「平滑的 argmax」

给定分数 $[2,1,0]$，softmax 输出大约 $[0.665,0.245,0.090]$：最高的那个拿到大部分预算，但其它也有份。
分数差距越大，分布越尖锐；差距越小，越接近均匀。

### 1.2 多头是「并行子空间」

把 $D$ 维拆成 $H$ 个 $d_k=D/H$ 维子空间，每个头在自己的子空间里做 attention。
最后把所有头的输出拼回 $D$ 维，再用 $W_O$ 混合。

> 💡 **一句话直觉**：**softmax = 把分数变成注意力预算；多头 = 用多组不同的 Q/K/V 投影，从多个角度同时看同一句话。**

类比边界：头之间不是完全独立的——它们共享输入，且 $W_O$ 会把它们混合；「每个头负责一种语法」是过度简化的说法。

## 2. 定义

### 2.1 softmax（按行）

对向量 $\mathbf{z}\in\mathbb{R}^{n}$，[softmax](#ref-bridle-1990) 定义为：

$$
\operatorname{softmax}(\mathbf{z})_i=\frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}
$$

对矩阵 $S\in\mathbb{R}^{T\times T}$，按**每一行**做 softmax：

$$
A_{ij}=\frac{e^{S_{ij}}}{\sum_{l=1}^{T}e^{S_{il}}}
$$

### 2.2 多头注意力

对第 $h$ 个头：

$$
Q_h=XW_Q^{(h)},\quad K_h=XW_K^{(h)},\quad V_h=XW_V^{(h)}
$$

$$
\text{head}_h=\operatorname{softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_k}}+M\right)V_h
$$

其中 $M$ 是 mask（合法位置 0，非法位置 $-\infty$）。

$$
\operatorname{MHA}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_H)W_O
$$

$W_O\in\mathbb{R}^{D\times D}$。

### 2.3 形状

| 张量 | 形状 |
|---|---|
| $X$ | `[B,T,D]` |
| $Q_h,K_h,V_h$ | `[B,H,T,d_k]` |
| $S_h$ | `[B,H,T,T]` |
| $A_h$ | `[B,H,T,T]` |
| $\text{head}_h$ | `[B,H,T,d_k]` |
| Concat | `[B,T,H·d_k]=[B,T,D]` |
| 输出 | `[B,T,D]` |

### 2.4 MHA / MQA / GQA

- **MHA**：每个头有独立的 K、V；$H_{kv}=H$。
- **MQA**（[Shazeer](#ref-shazeer-2019)）：所有头共享一组 K、V；$H_{kv}=1$。
- **GQA**（[Ainslie 等](#ref-ainslie-2023)）：每 $g$ 个 query 头共享一组 K、V；$1<H_{kv}<H$。

KV 参数量与缓存随 $H_{kv}$ 线性下降。

## 3. 为什么这样定义

**为什么 softmax 而不是直接归一化？** 直接除以和会让负分数变成负权重；指数保证非负，且放大差异，让模型能「聚焦」。指数还让 softmax 对整体平移不变（加常数不改变输出）。

**为什么多头？** 单个 softmax 只能产生一种「注意力分配」。多头提供多个分配，且每个头在不同子空间计算，能捕捉不同的关系。$W_O$ 再把它们组合。

**为什么拼接后还要 $W_O$？** 拼接只是把各头并排放置，$W_O$ 让不同头的信息交叉混合，而不是简单相加。

## 4. 性质与推导

### 4.1 softmax 的性质

- **非负、和为 1**：输出是概率分布；
- **平移不变**：$\operatorname{softmax}(\mathbf{z}+c)=\operatorname{softmax}(\mathbf{z})$；
- **温度**：$\operatorname{softmax}(\mathbf{z}/T)$，$T<1$ 更尖锐，$T>1$ 更平滑；
- **Jacobian**：$\dfrac{\partial \mathbf{p}}{\partial \mathbf{z}}=\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$。

### 4.2 数值稳定实现

直接算 $e^{z_i}$ 会上溢。标准做法：

$$
\operatorname{softmax}(\mathbf{z})_i=\frac{e^{z_i-\max_j z_j}}{\sum_j e^{z_j-\max_j z_j}}
$$

平移不改变结果（性质 4.1），但避免了 $e^{1000}$。

### 4.3 参数量

每个头：$W_Q,W_K,W_V$ 各 $D\times d_k$，共 $3Dd_k$；$H$ 个头合计 $3D^2$。加 $W_O$ 的 $D^2$，总共 $4D^2$。

**注意**：多头并不比单头（$d_k=D$）参数多；它只是把同样的 $D^2$ 拆成 $H$ 份。

### 4.4 计算复杂度

每个头 $O(T^2d_k)$，$H$ 个头合计 $O(T^2D)$；与单头相同量级。多头的主要代价在显存（$H$ 个 $T\times T$ 矩阵）和小的 GEMM 效率。

### 4.5 掩码

加性 mask：$S'=S+M$，$M_{ij}=0$ 或 $-\infty$。softmax 后非法位置权重为 0。因果 mask 使 $A$ 下三角。

## 5. 例子

### 5.1 手算：softmax 与温度

$\mathbf{z}=(2,1,0)$：

$$
e^2=7.389,\ e^1=2.718,\ e^0=1,\quad \text{和}=11.107
$$

$$
\operatorname{softmax}(\mathbf{z})=(0.665,0.245,0.090)
$$

温度 $T=0.5$：$\mathbf{z}/T=(4,2,0)$，$e^4=54.598$，$e^2=7.389$，$1$，和 $62.987$：

$$
(0.867,0.117,0.016)
$$

更尖锐。$T=2$：$(1,0.5,0)$，输出更均匀。

### 5.2 手算：两个头

设 $D=4$，$H=2$，$d_k=2$。头 1 的 $W_Q^{(1)}$ 取前两维，头 2 取后两维（仅示意）。
输入两个 token：$\mathbf{x}_1=(1,0,0,0)$，$\mathbf{x}_2=(0,1,0,0)$。
头 1 看到 $(1,0)$ 与 $(0,1)$，正交，注意力接近均匀；头 2 看到 $(0,0)$ 与 $(0,0)$，分数全 0，注意力完全均匀。
拼接后 $(0.5,0.5,0,0)$ 与 $(0.5,0.5,0,0)$ 再经 $W_O$ 混合——说明不同头可以关注不同子空间。

### 5.3 场景：GQA 省显存

Llama 2 70B 使用 GQA：$H=64$ 个 query 头，$H_{kv}=8$ 组 KV。KV Cache 约为 MHA 的 $8/64=1/8$。
代价是每组 KV 要服务 8 个 query 头，表达能力略降，但推理显存大幅下降。

## 6. 常见误区

1. **误区**：每个头负责一种可命名的语言现象。
   **纠正**：有些头确实表现出可解释模式，但大多数头功能重叠、可替代；头的重要性需通过消融验证。

2. **误区**：多头比单头参数多 $H$ 倍。
   **纠正**：总参数仍是 $4D^2$；只是把维度拆成 $H$ 份，每份 $d_k=D/H$。

3. **误区**：softmax 输出是「校准的概率」。
   **纠正**：它只是归一化分数，不保证置信度校准；温度、训练目标都会影响。

4. **误区**：attention 权重越大，该 token 越「重要」。
   **纠正**：权重是混合比例；某个 value 可能本身接近零，权重再大也不贡献信息。

## 7. 概念网络

- 前置：[11.3 QKV](03-Attention-上-QKV.md) —— 单头 attention。
- 前置：[9.2 多元高斯](../09-概率与统计/02-多元高斯.md) —— softmax 与分布。
- 后续：[11.7 KV Cache](07-KVCache.md) —— GQA 的显存收益。
- 后续：[11.11 采样](11-logits与采样.md) —— softmax 的另一个用途。
- 关联：[12.1 einsum](../12-进阶/01-张量与einsum.md) —— 多头注意力的统一写法。

## 8. 通往大模型：实现与代价

**PyTorch 形状**：

```python
# X: [B, T, D]
q = X @ Wq                      # [B, T, D]
q = q.view(B, T, H, dk).transpose(1, 2)   # [B, H, T, dk]
s = q @ k.transpose(-2, -1) / math.sqrt(dk)  # [B, H, T, T]
a = torch.softmax(s + mask, dim=-1)
out = (a @ v).transpose(1, 2).reshape(B, T, D) @ Wo
```

**FlashAttention**：不显式存储 $[B,H,T,T]$，而是分块计算 softmax 与加权和，把显存从 $O(T^2)$ 降到 $O(T)$；计算量仍是 $O(T^2)$。

**可检验结论**：因果 mask 下，$A$ 的每一行和为 1，且 $A_{ij}=0$（$j>i$）。用 `assert torch.allclose(a.sum(-1), torch.ones(...))` 可验证。

## 9. 练习

1. **概念**：softmax 为什么要减去最大值？不减会怎样？
2. **计算**：$\mathbf{z}=(1,1,1,1)$ 的 softmax 是什么？$\mathbf{z}=(10,0,0,0)$ 呢？
3. **计算**：$D=4096$，$H=32$，求 $d_k$；四个投影的总参数量。
4. **场景**：GQA 中 $H=32$、$H_{kv}=8$，KV Cache 是 MHA 的几分之几？代价是什么？
5. **开放**：如果所有头的 $W_Q,W_K,W_V$ 都相同，多头还有意义吗？为什么？

<details>
<summary>参考答案</summary>

1. 防止指数上溢；减去最大值后结果不变（平移不变性），但数值安全。
2. 均匀分布 $(0.25,0.25,0.25,0.25)$；$(0.9999,0.00003,\dots)$，极度尖锐。
3. $d_k=128$；$4D^2=4\times4096^2=67{,}108{,}864$。
4. $8/32=1/4$；每组 KV 服务 4 个 query 头，表达力略降，且实现更复杂。
5. 没有意义；所有头产生相同的注意力模式和输出，拼接后只是重复，$W_O$ 也无法创造新信息。多头的前提是**不同投影**。

</details>

## 10. 一页速记

- softmax 按行归一化；减最大值保证数值稳定。
- 多头 = $H$ 个 $d_k=D/H$ 子空间并行 attention，再拼接 + $W_O$。
- 参数量仍是 $4D^2$；计算量 $O(T^2D)$。
- MQA/GQA 减少 KV 头数，省显存。
- 因果 mask 在 softmax 前加 $-\infty$。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、方法名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-vaswani-2017"></a>**Vaswani 等（2017）**：*Attention Is All You Need*。多头注意力「拆成 $H$ 个 $d_k=D/H$ 子空间、拼接后再乘 $W_O$」的结构与 $4D^2$ 参数账都出自这里，对应本页第 2.2 与 4.3 节。[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- <a id="ref-bridle-1990"></a>**Bridle（1990）**：*Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition*。把前馈网络的输出经指数归一化解释成概率分布，softmax 这个名字与它的最大似然含义由此而来；本页的按行归一化、和为 1 就是它的直接应用。[DOI:10.1007/978-3-642-76153-9_28](https://doi.org/10.1007/978-3-642-76153-9_28)
- <a id="ref-ainslie-2023"></a>**Ainslie 等（2023）**：*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。让多个 query 头共享一组 K、V，并给出从 MHA 检查点「升级」到 GQA 的做法；本页第 2.4 与 5.3 节的 $H_{kv}$ 记法、KV Cache 按比例缩小都来自它。[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- <a id="ref-shazeer-2019"></a>**Shazeer（2019）**：*Fast Transformer Decoding: One Write-Head is All You Need*。MQA：解码时所有 query 头共用一份 K、V，把 KV Cache 压到 $1/H$，是本页 MQA/GQA 讨论的起点。[arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

---

下一篇 → [11.5 位置编码与 RoPE](05-位置编码与RoPE.md)
