---
id: 11.1
title: Token、Embedding 与隐藏状态
module: M11
slug: token-embedding
prereq: [1.1, 3.2]
minutes: 30
level: 核心
tags: [token, embedding, 隐藏状态, 词表]
llm_hooks: [embedding-table, hidden-state, logit-lens]
---

# 11.1 Token、Embedding 与隐藏状态

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[1.1](../01-向量与几何/01-向量是什么.md) ｜ **预计**：30 分钟
>
> **一句话**：模型先把离散符号查表变成向量，再让这个向量在几十层里不断被改写；「同一个词在不同句子里不同」发生在改写阶段，而不是查表阶段。

## 0. 场景引入

输入「猫坐在垫子上」，模型看到的第一样东西不是文字，而是四个整数，比如 `[8721, 4021, 9902, 318]`。

然后每个整数变成 4096 个浮点数。**这一步叫 embedding。**

但问题来了：如果「猫」永远查同一行，模型怎么区分「猫坐在垫子上」和「猫在键盘上」里的猫？

答案：查表给出的是**初始**向量；后续每一层的 attention 会把它和上下文混合，得到**上下文相关**的隐藏状态。

## 1. 直觉

把 embedding 表想成一本**字典**：

- 字典有 $V$ 页（词表大小）；
- 每页写着一行 $D$ 个数（该词的向量）；
- 查一个词，就是取出对应那一行。

查表之后，每个 token 都变成一个箭头。句子就变成一串箭头，排成 `[T, D]`；一批句子就是 `[B, T, D]`。

> 💡 **一句话直觉**：**embedding = 把离散 ID 翻译成连续向量；隐藏状态 = 这个向量被上下文改写后的结果。**

类比边界：embedding 的每一维**没有**天然语义（不像「第 3 维代表动物性」）；语义分布在方向的组合里。

## 2. 定义

### 2.1 词表与 token

分词器把文本切成 token，并映射为整数 ID：

$$
\text{文本}\ \xrightarrow{\text{tokenizer}}\ (t_1,\dots,t_T),\qquad t_i\in\{0,\dots,V-1\}
$$

$V$ 是词表大小。token 不一定是词，可能是子词、字符或字节。

### 2.2 Embedding 表

$$
E\in\mathbb{R}^{V\times D}
$$

第 $k$ 行 $E[k,:]$ 是 token $k$ 的向量。查表：

$$
\mathbf{x}_i=E[t_i,:]\in\mathbb{R}^{D}
$$

对 batch：

$$
X=E[T]\in\mathbb{R}^{B\times T\times D}
$$

其中 $T\in\mathbb{N}^{B\times T}$ 是 ID 矩阵（这里用 $T$ 同时表示序列长度，注意上下文）。

### 2.3 隐藏状态

经过第 $l$ 层后的表示：

$$
H^{(l)}\in\mathbb{R}^{B\times T\times D}
$$

$H^{(0)}=X+(\text{位置信息})$。最终层 $H^{(L)}$ 用于预测下一个 token。

### 2.4 与 one-hot 的等价

若 $\mathbf{e}_k\in\mathbb{R}^V$ 是 one-hot，则

$$
E^\top\mathbf{e}_k=E[k,:]^\top
$$

**查表 = one-hot 向量乘 embedding 矩阵**。这解释了为什么 embedding 层本质上是一个线性层，只是输入是 one-hot、计算被优化成索引。

## 3. 为什么这样定义

离散符号无法直接做加减、求导。embedding 把符号映射到连续空间，带来三个好处：

1. **可学习**：向量随梯度更新，模型自己决定「猫」应该靠近「狗」还是「汽车」；
2. **可组合**：向量能相加、投影、算相似度；
3. **可微**：从损失到 embedding 的梯度可以回传（尽管查表只更新被用到的行）。

为什么不用 one-hot 直接喂给模型？因为 one-hot 维度是 $V$（可达几十万），且任意两个词正交，无法表达相似性。embedding 把维度压到 $D$，并让相似词共享方向。

## 4. 性质与推导

### 4.1 参数量

$$
\#\text{params}=V\times D
$$

对 $V=128000,D=4096$：约 5.24 亿。小模型里 embedding 占比很高；大模型里占比下降。

### 4.2 权重共享（weight tying）

输出层把隐藏状态映射到词表 logits：

$$
\text{logits}=H W_{\text{vocab}},\qquad W_{\text{vocab}}\in\mathbb{R}^{D\times V}
$$

若令 $W_{\text{vocab}}=E^\top$，则输入 embedding 与输出 unembedding 共享参数，省下 $V\times D$ 参数。很多语言模型这样做。

### 4.3 位置信息

纯 embedding 没有顺序感：「猫追狗」和「狗追猫」的 token 集合相同。因此要加位置编码（第 11.5 节）：

$$
H^{(0)}=E[T]+P
$$

其中 $P\in\mathbb{R}^{T\times D}$ 是位置编码（可学习或固定）。

### 4.4 上下文相关

同一 token 在不同位置的初始向量相同；经过 attention 后，

$$
\mathbf{h}_i^{(l)}=\mathbf{h}_i^{(l-1)}+\sum_j \alpha_{ij}W_V\mathbf{h}_j^{(l-1)}
$$

其中 $\alpha_{ij}$ 依赖整句。因此「猫」在「猫坐在垫子上」与「猫在键盘上」中的最终表示不同。

## 5. 例子

### 5.1 手算：小词表查表

设 $V=5$，$D=3$，词表 `["<pad>", "猫", "狗", "汽车", "苹果"]`，

$$
E=\begin{bmatrix}
0&0&0\\
0.9&0.1&0.1\\
0.8&0.2&0.1\\
0.1&0.9&0.1\\
0.1&0.1&0.9
\end{bmatrix}
$$

句子「猫 狗」的 ID 是 $[1,2]$，查表得

$$
X=\begin{bmatrix}0.9&0.1&0.1\\0.8&0.2&0.1\end{bmatrix}\in\mathbb{R}^{2\times3}
$$

### 5.2 场景：一个 token 的旅程

```text
"猫" → id 8721 → E[8721] = x ∈ R^4096
     → + 位置编码
     → 32 层 × (attention + FFN + 残差)
     → h ∈ R^4096
     → h @ W_vocab → logits ∈ R^128000
     → softmax → 下一个 token 的概率
```

**注意**：$E$ 只在第 0 层使用；后面每层都有新的权重矩阵，但隐藏维度始终是 $D$。

### 5.3 场景：logit lens

把中间层 $H^{(l)}$ 直接乘以 unembedding，可以得到「这一层倾向于预测什么」。这是一种粗略的可解释性工具，前提是各层表示空间大致对齐——这个前提并不总成立，所以结果只能当线索。

## 6. 常见误区

1. **误区**：embedding 的某一维对应某个语义特征。
   **纠正**：通常不是。语义分布在多个维度的组合方向上；单维往往不可解释。

2. **误区**：同一个词的 embedding 在所有语境下相同。
   **纠正**：**初始** embedding 相同，最终隐藏状态不同。

3. **误区**：embedding 参数量 = 模型参数量的大部分。
   **纠正**：小模型可能如此；大模型中 $12D^2L$ 主导，embedding 占比下降。

4. **误区**：token 就是词。
   **纠正**：token 可能是子词/字节；一个词可能被切成多个 token，这会影响模型对拼写、罕见词的处理。

## 7. 概念网络

- 前置：[1.1 向量](../01-向量与几何/01-向量是什么.md) —— 向量是数据点。
- 前置：[3.2 矩阵乘向量](../03-矩阵与线性映射/02-矩阵乘向量.md) —— 查表 = one-hot 乘矩阵。
- 后续：[11.2 线性层](02-线性层.md) —— 隐藏状态如何被变换。
- 后续：[11.3 Attention](03-Attention-上-QKV.md) —— 上下文如何混合。
- 后续：[11.11 logits 与采样](11-logits与采样.md) —— 隐藏状态如何变回词。

## 8. 通往大模型：形状与参数

**形状链**：

$$
[B,T]\xrightarrow{E}[B,T,D]\xrightarrow{+P}[B,T,D]
$$

**参数量**：$V\times D$（可选择性绑定输出层）。

**可检验结论**：若两个 token 的 ID 相同，它们的初始 embedding 逐元素相等；若不同，则不一定正交（取决于训练）。对随机初始化的 $E$，两行近似正交；训练后相似词会靠近。

```python
import numpy as np
V, D, B, T = 5, 3, 2, 4
E = np.random.randn(V, D)
ids = np.array([[1, 2, 1, 0], [3, 4, 0, 0]])
X = E[ids]                 # [B, T, D]
print(X.shape)             # (2, 4, 3)
print(np.allclose(X[0, 0], E[1]))  # True
```

## 9. 练习

1. **概念**：为什么 embedding 比 one-hot 更适合表达语义相似？
2. **计算**：$V=32000$，$D=1024$，embedding 参数量是多少？若输出层绑定权重，总共省了多少参数？
3. **计算**：一个 batch 有 $B=4$、$T=512$、$D=768$，隐藏状态张量有多少个元素？fp16 下占多少 MiB？
4. **场景**：为什么「同一个词在不同句子中表示不同」不能由 embedding 表解释？
5. **开放**：如果词表里有 10 万个 token，但实际训练只用到 5 万个，未使用 token 的 embedding 会怎样？有什么风险？

<details>
<summary>参考答案</summary>

1. one-hot 两两正交，无法表达相似；embedding 是低维连续向量，可通过训练让相似词方向接近。
2. $32000\times1024=32{,}768{,}000$；绑定后省同样的 3276.8 万（输入输出共享一份）。
3. $4\times512\times768=1{,}572{,}864$ 个元素；fp16 约 $3$ MiB。
4. embedding 表是静态查表，同一 ID 永远同一行；上下文差异来自 attention/FFN 对隐藏状态的改写。
5. 未使用 token 的 embedding 可能停留在初始化附近，若被采样到会产生无意义输出；也可能被 weight decay 拉向零。风险是罕见 token 表示质量差。

</details>

## 10. 一页速记

- token → ID → 查 $E\in\mathbb{R}^{V\times D}$ → `[B,T,D]`。
- 查表 = one-hot 乘 $E$；参数量 $VD$。
- 初始 embedding 与上下文无关；隐藏状态与上下文相关。
- 位置编码在 embedding 之后加入。
- 输出层可绑定 $E^\top$，省 $VD$ 参数。

---

下一篇 → [11.2 线性层：一次矩阵乘法](02-线性层.md)
