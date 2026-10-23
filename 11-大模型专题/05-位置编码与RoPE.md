---
id: 11.5
title: 位置编码与 RoPE 的旋转矩阵
module: M11
slug: rope
prereq: [1.6, 6.4]
minutes: 35
level: 核心
tags: [位置编码, RoPE, 旋转矩阵, 相对位置]
llm_hooks: [rope, long-context, position]
---

# 11.5 位置编码与 RoPE 的旋转矩阵

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[1.6](../01-向量与几何/06-正交与投影.md) ｜ **预计**：35 分钟
>
> **一句话**：attention 本身不知道词的顺序；RoPE 用「按位置旋转 Q/K」的方式把相对位置写进点积里，而且不增加任何参数。

## 0. 场景引入

「猫追狗」和「狗追猫」的 token 集合相同，attention 若只看内容，输出会一样——这显然不对。

早期 Transformer 用「位置编码相加」：$H^{(0)}=E[T]+P$，$P$ 是位置向量。
但相加式位置编码在长上下文外推时表现一般，而且它把位置信息混进了内容向量。

RoPE（Rotary Position Embedding）换了个思路：**不改变向量内容，而是按位置旋转 query/key**。两个 token 的点积只依赖它们的**相对位置**。

## 1. 直觉

想象每个 token 的 Q/K 是一根指针。位置 0 不旋转，位置 1 转 $\theta$，位置 2 转 $2\theta$……

当位置 $m$ 的 query 与位置 $n$ 的 key 做点积时，两者的夹角里包含 $(m-n)\theta$。**绝对位置被旋转「抵消」了，只剩相对差。**

> 💡 **一句话直觉**：**RoPE = 按位置旋转 Q/K；点积自动变成「相对位置的函数」。**

类比边界：旋转是正交变换，保长度、保夹角；它不改变向量的「内容」，只改变方向。不同维度对（频率）可以有不同的 $\theta$。

## 2. 定义

### 2.1 二维旋转

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
$$

性质：$R(\theta)^\top R(\phi)=R(\phi-\theta)$，$\det R=1$，$R^\top=R^{-1}$。

### 2.2 RoPE 的频率

把 $d_k$ 维分成 $d_k/2$ 个二维对。第 $i$ 对的旋转角：

$$
\theta_i=\text{base}^{-2i/d_k},\qquad i=0,1,\dots,d_k/2-1
$$

位置 $m$ 的总旋转：

$$
R_m=\operatorname{diag}\big(R(m\theta_0),R(m\theta_1),\dots\big)
$$

对 query/key：

$$
\mathbf{q}_m'=R_m\mathbf{q}_m,\qquad \mathbf{k}_n'=R_n\mathbf{k}_n
$$

### 2.3 相对位置性质

$$
(\mathbf{q}_m')^\top\mathbf{k}_n'
=\mathbf{q}_m^\top R_m^\top R_n\mathbf{k}_n
=\mathbf{q}_m^\top R_{n-m}\mathbf{k}_n
$$

右边只依赖 $n-m$，不依赖绝对位置 $m,n$。

### 2.4 实现形式

对每一对 $(x_{2i},x_{2i+1})$：

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=\begin{bmatrix}\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
$$

通常只对 Q、K 应用，不对 V 应用。

## 3. 为什么这样定义

**为什么旋转而不是相加？**

- 旋转是正交变换，不改变向量长度，数值更稳定；
- 点积自然产生相对位置项，不需要额外的相对位置表；
- 没有可学习参数，不增加模型大小；
- 不同频率的对可以编码不同尺度的位置（类似傅里叶特征）。

**为什么只对 Q/K？** 位置影响的是「谁该关注谁」，即匹配分数；V 是内容，不需要旋转。若旋转 V，输出会被位置「污染」。

## 4. 性质与推导

### 4.1 正交性

$R_m^\top R_m=I$，所以 $\|\mathbf{q}_m'\|=\|\mathbf{q}_m\|$。RoPE 不改变 Q/K 的范数，只改变方向。

### 4.2 相对位置

上面的推导：

$$
R_m^\top R_n=R(-m\theta)R(n\theta)=R((n-m)\theta)
$$

因此内积只依赖 $n-m$。

### 4.3 远程衰减

不同频率的旋转叠加，使远距离位置的内积倾向于更小（类似「相对位置越远，相关性越弱」的先验）。这不是硬约束，而是频率设计带来的偏好。

### 4.4 外推与插值

训练长度之外，$\theta_i$ 与 $m$ 的组合可能超出训练分布。常见做法：位置插值（把 $m$ 缩放）、NTK-aware 调整 base、YaRN 等。它们改的是频率/位置映射，不改变 RoPE 的代数结构。

### 4.5 计算

对每个 token、每个 pair 做一次 2×2 旋转：$O(Td_k)$，相比 attention 的 $O(T^2d_k)$ 可忽略。

## 5. 例子

### 5.1 手算：二维旋转

设 $\mathbf{q}=(1,0)$，位置 $m=1$，$\theta=90^\circ$：

$$
\mathbf{q}'=R(90^\circ)\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}0\\1\end{bmatrix}
$$

设 $\mathbf{k}=(1,0)$，位置 $n=2$，则 $\mathbf{k}'=R(180^\circ)(1,0)=(-1,0)$。

$$
(\mathbf{q}')^\top\mathbf{k}'=0\cdot(-1)+1\cdot0=0
$$

用相对位置公式：$R_{n-m}=R(90^\circ)$，$\mathbf{q}^\top R(90^\circ)\mathbf{k}=(1,0)^\top(0,1)=0$，一致。

### 5.2 手算：两个频率

$d_k=4$，$\text{base}=10000$：

$$
\theta_0=1,\qquad \theta_1=10000^{-2/4}=0.01
$$

位置 $m=10$：第一对转 $10$ rad（约 $573^\circ$），第二对转 $0.1$ rad。
高频对区分近距离，低频对区分远距离——这就是「多尺度」。

### 5.3 场景：为什么 RoPE 不增加参数

位置编码若用可学习表，长度 $T_{\max}\times D$ 个参数，且无法外推。RoPE 只有固定的三角函数，参数量为 0，长度由 $\theta_i$ 与位置计算决定。

## 6. 常见误区

1. **误区**：RoPE 是把位置向量加到 embedding 上。
   **纠正**：RoPE 是**旋转** Q/K，不是相加；相加式位置编码是另一类方法。

2. **误区**：RoPE 对 V 也旋转。
   **纠正**：通常只旋转 Q、K；旋转 V 会改变内容表示，破坏「value 传递内容」的语义。

3. **误区**：RoPE 编码的是绝对位置。
   **纠正**：虽然旋转角由绝对位置决定，但点积只依赖相对位置；这正是它的设计目的。

4. **误区**：RoPE 可以无限外推。
   **纠正**：代数上相对位置仍成立，但频率与训练分布不匹配会导致性能下降；需要插值/调整 base 等方法。

## 7. 概念网络

- 前置：[1.6 正交与投影](../01-向量与几何/06-正交与投影.md) —— 旋转保长度。
- 前置：[6.4 谱定理](../06-特征值/04-对称矩阵与谱定理.md) —— 正交矩阵的结构。
- 后续：[11.7 KV Cache](07-KVCache.md) —— 缓存的是旋转后的 K。
- 关联：[11.3 Attention](03-Attention-上-QKV.md) —— RoPE 作用于 Q/K。

## 8. 通往大模型：形状与实现

**形状**：RoPE 不改变形状，只改变数值：

$$
Q,K:[B,H,T,d_k]\ \longrightarrow\ Q',K':[B,H,T,d_k]
$$

**实现（成对旋转）**：

```python
def rope(x, positions, base=10000.0):
    # x: [..., T, d_k]
    d = x.shape[-1]
    i = np.arange(0, d, 2)
    theta = base ** (-i / d)             # [d/2]
    ang = positions[..., None] * theta   # [..., T, d/2]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
```

**可检验结论**：对任意位置 $m,n$，$\langle R_m\mathbf{q},R_n\mathbf{k}\rangle$ 只依赖 $n-m$。可固定 $\mathbf{q},\mathbf{k}$，改变 $m,n$ 但保持差不变，验证内积不变。

## 9. 练习

1. **概念**：RoPE 为什么不增加参数？它和可学习位置编码的本质区别是什么？
2. **计算**：$R(30^\circ)$ 作用于 $(1,0)$，结果是什么？
3. **计算**：验证 $R(\alpha)^\top R(\beta)=R(\beta-\alpha)$（用 $2\times2$ 矩阵乘法）。
4. **场景**：如果只对 Q 旋转、不对 K 旋转，相对位置性质还成立吗？
5. **开放**：长上下文外推时，为什么「调整 base」可能有效？它改变了什么？

<details>
<summary>参考答案</summary>

1. 角度由位置和固定频率算出，无学习参数；可学习编码有一张 $T_{\max}\times D$ 表，且不能自然外推。
2. $(\cos30^\circ,\sin30^\circ)\approx(0.866,0.5)$。
3. 用和角公式展开即可；几何上先转 $\beta$ 再转 $-\alpha$，净旋转 $\beta-\alpha$。
4. 不成立；$R_m^\top$ 只作用于 Q，内积会依赖 $m$ 和 $n$ 的绝对位置组合。
5. base 控制频率间隔；增大 base 让低频变化更慢，使训练长度外的位置角变化更平缓，从而缓解分布外推问题，但不改变 RoPE 的代数形式。

</details>

## 10. 一页速记

- RoPE：对 Q/K 按位置旋转，$\theta_i=\text{base}^{-2i/d_k}$。
- $R_m^\top R_n=R_{n-m}$，点积只依赖相对位置。
- 正交、保范数、零参数、只作用于 Q/K。
- 形状不变；实现按二维对旋转。
- 外推需插值/调 base，代数性质不保证泛化。

---

下一篇 → [11.6 LayerNorm 与 RMSNorm 的几何](06-LayerNorm与RMSNorm.md)
