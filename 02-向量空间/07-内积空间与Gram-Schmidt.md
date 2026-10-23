---
id: 2.7
title: 内积空间、正交基与 Gram–Schmidt
module: M02
slug: inner-product-gram-schmidt
prereq: [1.6, 2.4]
minutes: 40
level: 核心
tags: [内积, 正交基, Gram-Schmidt, QR分解]
llm_hooks: [orthogonal-init, qr-decomposition, decorrelation]
---

# 2.7 内积空间、正交基与 Gram–Schmidt

> **本节位置**：模块 2 · 向量空间 ｜ **前置**：[1.6](../01-向量与几何/06-正交与投影.md)、[2.4](04-基与维数.md) ｜ **预计**：40 分钟
>
> **一句话**：给向量空间加上内积，就有了长度和角度；再把基变成正交的，投影就只需一次次点积。

## 0. 场景引入

你已经会求一个向量在一条直线上的投影。现在要投影到一个**平面**上，平面由两个斜着的基向量张成。

直接套公式会发现：两个基向量不正交，投影会「重复计算」重叠分量。怎么办？

答案：先把这组基**正交化**（Gram–Schmidt），再分别投影、相加。这正是 QR 分解与最小二乘数值算法的核心。

## 1. 直觉

正交基就像一套**互相垂直的坐标轴**：

- 想知道向量在某个轴上的分量？只需一次点积；
- 轴之间没有重叠，不用「扣除」彼此的影响；
- 长度满足勾股定理：$\|\mathbf{x}\|^2=\sum_i\langle\mathbf{x},\mathbf{q}_i\rangle^2$。

Gram–Schmidt 的过程就是：拿第一根轴，第二根减去在第一根上的投影，第三根减去在前两根上的投影……每一根都「擦掉」已有方向，只留下新方向。

> 💡 **一句话直觉**：**Gram–Schmidt = 逐个「减去已有方向的影子」，得到互相垂直的新基。**

## 2. 定义

### 2.1 内积空间

实向量空间 $V$ 上若定义了内积 $\langle\cdot,\cdot\rangle$（对称、双线性、正定，见第 1.3 节），称 $(V,\langle\cdot,\cdot\rangle)$ 为**内积空间（inner product space）**。

内积诱导范数 $\|\mathbf{x}\|=\sqrt{\langle\mathbf{x},\mathbf{x}\rangle}$，进而诱导距离与夹角。

### 2.2 正交基与标准正交基

- **正交基**：基向量两两正交，$\langle\mathbf{b}_i,\mathbf{b}_j\rangle=0$（$i\neq j$）；
- **标准正交基（orthonormal basis）**：还要求 $\|\mathbf{b}_i\|=1$，即

$$
\langle\mathbf{q}_i,\mathbf{q}_j\rangle=\delta_{ij}
=\begin{cases}1,&i=j\\0,&i\neq j\end{cases}
$$

### 2.3 Gram–Schmidt 正交化

给定线性无关的 $\mathbf{v}_1,\dots,\mathbf{v}_k$：

$$
\begin{aligned}
\mathbf{u}_1&=\mathbf{v}_1, & \mathbf{q}_1&=\frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}\\
\mathbf{u}_2&=\mathbf{v}_2-\langle\mathbf{v}_2,\mathbf{q}_1\rangle\mathbf{q}_1, & \mathbf{q}_2&=\frac{\mathbf{u}_2}{\|\mathbf{u}_2\|}\\
\mathbf{u}_3&=\mathbf{v}_3-\langle\mathbf{v}_3,\mathbf{q}_1\rangle\mathbf{q}_1-\langle\mathbf{v}_3,\mathbf{q}_2\rangle\mathbf{q}_2, & \mathbf{q}_3&=\frac{\mathbf{u}_3}{\|\mathbf{u}_3\|}\\
&\vdots
\end{aligned}
$$

一般地：

$$
\mathbf{u}_j=\mathbf{v}_j-\sum_{i=1}^{j-1}\langle\mathbf{v}_j,\mathbf{q}_i\rangle\mathbf{q}_i
$$

### 2.4 QR 分解

若 $A\in\mathbb{R}^{m\times n}$ 的列线性无关，则可分解为

$$
A=QR
$$

其中 $Q\in\mathbb{R}^{m\times n}$ 的列标准正交，$R\in\mathbb{R}^{n\times n}$ 上三角且对角元为正。

## 3. 为什么这样定义

在普通基下，求坐标要解线性方程组（$P^{-1}\mathbf{v}$）；在正交基下，坐标直接是内积：

$$
\mathbf{v}=\sum_i\langle\mathbf{v},\mathbf{q}_i\rangle\mathbf{q}_i
$$

**证明**：两边与 $\mathbf{q}_j$ 做内积，由正交性只剩 $j$ 项，得 $\langle\mathbf{v},\mathbf{q}_j\rangle=\langle\mathbf{v},\mathbf{q}_j\rangle$。

这把「解方程」降级为「做点积」，是数值线性代数偏爱正交基的根本原因。正交矩阵还保范数、保内积，误差不会被放大（第 10.2 节）。

## 4. 性质与推导

### 4.1 Gram–Schmidt 为什么有效

归纳证明：假设 $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$ 已标准正交。令

$$
\mathbf{u}_j=\mathbf{v}_j-\sum_{i<j}\langle\mathbf{v}_j,\mathbf{q}_i\rangle\mathbf{q}_i
$$

对任意 $l<j$：

$$
\langle\mathbf{u}_j,\mathbf{q}_l\rangle
=\langle\mathbf{v}_j,\mathbf{q}_l\rangle-\sum_{i<j}\langle\mathbf{v}_j,\mathbf{q}_i\rangle\langle\mathbf{q}_i,\mathbf{q}_l\rangle
=\langle\mathbf{v}_j,\mathbf{q}_l\rangle-\langle\mathbf{v}_j,\mathbf{q}_l\rangle=0
$$

所以 $\mathbf{u}_j$ 与所有已有 $\mathbf{q}_l$ 正交。$\mathbf{u}_j\neq\mathbf{0}$ 是因为 $\mathbf{v}_j$ 不在前 $j-1$ 个向量的张成中（原组线性无关）。

### 4.2 QR 与投影

$A=QR$ 中，$Q$ 的列是 $A$ 的列空间的标准正交基。投影矩阵

$$
P=QQ^\top
$$

最小二乘解可写成 $R\hat{\mathbf{c}}=Q^\top\mathbf{y}$，用回代求解，避免显式求 $A^\top A$ 的逆（数值更稳）。

### 4.3 计算复杂度

对 $m\times n$ 矩阵，经典 Gram–Schmidt 约 $2mn^2$ 次浮点运算；改进版（MGS）数值更稳，但仍有累积误差。实际库用 Householder 变换（第 3.5 节）。

## 5. 例子

### 5.1 手算：$\mathbb{R}^3$ 中的 Gram–Schmidt

$\mathbf{v}_1=(1,1,0)$，$\mathbf{v}_2=(1,0,1)$，$\mathbf{v}_3=(0,1,1)$。

**第一步**：

$$
\mathbf{q}_1=\frac{(1,1,0)}{\sqrt2}=\left(\frac{1}{\sqrt2},\frac{1}{\sqrt2},0\right)
$$

**第二步**：$\langle\mathbf{v}_2,\mathbf{q}_1\rangle=\frac{1}{\sqrt2}$，

$$
\mathbf{u}_2=(1,0,1)-\frac{1}{\sqrt2}\left(\frac{1}{\sqrt2},\frac{1}{\sqrt2},0\right)
=\left(\frac12,-\frac12,1\right)
$$

$$
\|\mathbf{u}_2\|=\sqrt{\frac14+\frac14+1}=\sqrt{\frac32},\qquad
\mathbf{q}_2=\left(\frac{1}{\sqrt6},-\frac{1}{\sqrt6},\frac{2}{\sqrt6}\right)
$$

**第三步**：$\langle\mathbf{v}_3,\mathbf{q}_1\rangle=\frac{1}{\sqrt2}$，$\langle\mathbf{v}_3,\mathbf{q}_2\rangle=\frac{1}{\sqrt6}$，

$$
\mathbf{u}_3=(0,1,1)-\frac12(1,1,0)-\frac16(1,-1,2)
=\left(-\frac23,\frac23,\frac23\right)
$$

$$
\|\mathbf{u}_3\|=\sqrt{\frac{12}{9}}=\frac{2}{\sqrt3},\qquad
\mathbf{q}_3=\left(-\frac{1}{\sqrt3},\frac{1}{\sqrt3},\frac{1}{\sqrt3}\right)
$$

验证：$\mathbf{q}_1^\top\mathbf{q}_3=0$，$\mathbf{q}_2^\top\mathbf{q}_3=\frac{-1-1+2}{\sqrt{18}}=0$。

### 5.2 场景：QR 解最小二乘

对第 1.6 节的拟合问题，$A=QR$，则

$$
A^\top A\hat{\mathbf{c}}=A^\top\mathbf{y}
\ \Longrightarrow\
R^\top Q^\top QR\hat{\mathbf{c}}=R^\top Q^\top\mathbf{y}
\ \Longrightarrow\
R\hat{\mathbf{c}}=Q^\top\mathbf{y}
$$

只需回代，不用求逆。这是 `numpy.linalg.lstsq` 背后的思路之一。

## 6. 常见误区

1. **误区**：Gram–Schmidt 只是「把向量变单位长度」。
   **纠正**：它还负责**正交化**；只归一化不改变夹角。

2. **误区**：正交化结果与输入顺序无关。
   **纠正**：顺序不同，得到的正交基通常不同（但张成空间相同）。

3. **误区**：经典 Gram–Schmidt 在计算机里总能得到正交基。
   **纠正**：浮点误差会累积，向量接近相关时尤其严重；实际用改进版或 Householder。

4. **误区**：QR 分解唯一。
   **纠正**：若规定 $R$ 对角元为正，则「薄 QR」唯一；否则可差一个对角符号矩阵。

## 7. 概念网络

- 前置：[1.6 正交与投影](../01-向量与几何/06-正交与投影.md) —— 减去投影就是正交化。
- 前置：[2.4 基与维数](04-基与维数.md) —— 正交基也是一种基。
- 后续：[2.8 正交补与投影定理](08-正交补与投影定理.md) —— 正交分解。
- 后续：[4.4 最小二乘](../04-线性方程组/04-最小二乘.md) —— QR 是稳定解法。
- 关联：10.2 条件数 —— 正交变换不放大误差。

## 8. 通往大模型：正交性为什么值钱

**1. 正交初始化**：让权重矩阵的列近似标准正交，使

$$
\|W\mathbf{x}\|\approx\|\mathbf{x}\|
$$

深层连乘时范数既不爆炸也不消失。QR 分解是构造正交矩阵的标准方法。

**2. 正交微调**：一些方法约束权重更新为正交矩阵，以保持预训练知识、减少遗忘；QR/极分解是常见工具。

**3. 头去相关**：多头注意力希望不同头关注不同模式；可以用正交约束或对头输出做去相关，减少冗余（第 2.3 节）。

**可检验结论**：若 $Q$ 正交，则 $\|Q\mathbf{x}\|_2=\|\mathbf{x}\|_2$，且 $\kappa_2(Q)=1$——这是条件数最好的矩阵，误差不会被放大。

## 9. 练习

1. **概念**：为什么在正交基下求坐标只需点积，而在普通基下要解方程？
2. **计算**：对 $\mathbf{v}_1=(1,0)$、$\mathbf{v}_2=(1,1)$ 做 Gram–Schmidt。
3. **计算**：对 $\mathbf{v}_1=(1,1,1)$、$\mathbf{v}_2=(1,0,1)$ 做 Gram–Schmidt（只需前两个）。
4. **场景**：为什么最小二乘用 QR 比直接求 $(A^\top A)^{-1}$ 更稳？
5. **开放**：如果两个输入向量几乎共线，Gram–Schmidt 会发生什么？如何在实现中检测？

<details>
<summary>参考答案</summary>

1. 正交基下 $\mathbf{v}=\sum c_i\mathbf{q}_i$，两边点乘 $\mathbf{q}_j$ 得 $c_j=\langle\mathbf{v},\mathbf{q}_j\rangle$；普通基下基向量互相影响，需解线性方程组。
2. $\mathbf{q}_1=(1,0)$；$\mathbf{u}_2=(1,1)-1\cdot(1,0)=(0,1)$，$\mathbf{q}_2=(0,1)$。
3. $\mathbf{q}_1=(1,1,1)/\sqrt3$；$\langle\mathbf{v}_2,\mathbf{q}_1\rangle=2/\sqrt3$；$\mathbf{u}_2=(1,0,1)-\frac23(1,1,1)=(\frac13,-\frac23,\frac13)$；$\|\mathbf{u}_2\|=\sqrt{6}/3$；$\mathbf{q}_2=(\frac{1}{\sqrt6},-\frac{2}{\sqrt6},\frac{1}{\sqrt6})$。
4. $A^\top A$ 会把条件数平方（$\kappa(A^\top A)=\kappa(A)^2$），放大舍入误差；QR 只涉及正交变换，条件数不放大。
5. 第二个向量减去投影后接近零，归一化会放大噪声；检测 $\|\mathbf{u}_j\|$ 小于阈值，或改用带列主元的 QR/Householder。

</details>

## 10. 一页速记

- 内积空间 = 向量空间 + 内积（长度、角度）。
- 标准正交基：$\langle\mathbf{q}_i,\mathbf{q}_j\rangle=\delta_{ij}$。
- 正交基下坐标 = 点积；$\mathbf{v}=\sum\langle\mathbf{v},\mathbf{q}_i\rangle\mathbf{q}_i$。
- Gram–Schmidt：逐个减去已有方向的投影。
- $A=QR$：$Q$ 正交，$R$ 上三角；最小二乘用 $R\hat{\mathbf{c}}=Q^\top\mathbf{y}$。

---

下一篇 → [2.8 正交补与投影定理](08-正交补与投影定理.md)
