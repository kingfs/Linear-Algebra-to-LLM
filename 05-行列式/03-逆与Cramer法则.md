---
id: 5.3
title: 逆、Cramer 法则与伴随矩阵
module: M05
slug: inverse-cramer-adjugate
prereq: [3.4, 5.2]
minutes: 30
level: 核心
tags: [逆矩阵, 伴随矩阵, Cramer法则, 可逆性]
llm_hooks: [gaussian-likelihood, normalizing-flow]
---

# 5.3 逆、Cramer 法则与伴随矩阵

> **本节位置**：模块 5 · 行列式 ｜ **前置**：[3.4 逆矩阵](../03-矩阵与线性映射/04-逆矩阵与可逆性.md)、[5.2](02-行列式的性质与计算.md) ｜ **预计**：30 分钟
>
> **一句话**：把余子式转置拼成伴随矩阵，再除以行列式，就得到逆；Cramer 法则只是这个公式逐分量写开。

## 0. 场景引入

解方程组 $3x+y=5,\ x+2y=5$。消元两步就能得到 $x=1,\ y=2$。但如果我们想要一个**只含行列式的闭式公式**，直接看出解对系数的依赖，该怎么办？

更一般地：给定 $A\in\mathbb{R}^{n\times n}$ 和 $\mathbf{b}$，能否用 $\det$ 判断 $A$ 是否可逆，并把 $A^{-1}\mathbf{b}$ 写成 $n$ 个行列式的比值？

## 1. 直觉

逆矩阵（inverse matrix）是「撤销变换」。在 5.1 里，$A$ 把体积放大 $|\det A|$ 倍；那么 $A^{-1}$ 就该把它缩回去，所以分母出现 $\det A$ 并不意外。

伴随矩阵（adjugate）$\operatorname{adj}(A)$ 可以理解为「余子式的转置」：每个元素记录「删掉某行某列后，剩下的变换把体积放大几倍」。把 $\operatorname{adj}(A)$ 除以 $\det A$，体积因子正好抵消。

Cramer 法则（Cramer's rule）则是逐分量的版本：要求 $x_i$，就把 $A$ 的第 $i$ 列换成 $\mathbf{b}$，算一个行列式，再除以 $\det A$。

> 💡 **一句话直觉**：**逆 = 伴随 / 行列式；Cramer = 用「替换一列后的体积」读出每个坐标**。

类比边界：这个直觉只在 $A$ 可逆时成立；$\det A=0$ 时体积已经被压成 0，没有「缩回去」的操作。

## 2. 定义

### 2.1 余子式与代数余子式

删掉 $a_{ij}$ 所在的行、列，得到 $(n-1)\times(n-1)$ 的**余子式（minor）** $M_{ij}$；带符号的 $C_{ij}=(-1)^{i+j}M_{ij}$ 叫**代数余子式（cofactor）**。按行展开：$\det A=\sum_j a_{ij}C_{ij}$。

### 2.2 伴随矩阵

**伴随矩阵（adjugate）**定义为代数余子式矩阵的转置：$\operatorname{adj}(A)_{ij}=C_{ji}=(-1)^{i+j}M_{ji}$。注意下标是 $ji$ 不是 $ij$：第 $i$ 行第 $j$ 列的元素，来自「删掉第 $j$ 行第 $i$ 列」的余子式。

### 2.3 逆矩阵公式

若 $\det A\neq0$，则 $A^{-1}=\frac{1}{\det A}\operatorname{adj}(A)$。

### 2.4 Cramer 法则

设 $A\mathbf{x}=\mathbf{b}$，$A$ 可逆。把 $A$ 的第 $i$ 列换成 $\mathbf{b}$，记作 $A_i$，则 $x_i=\det A_i/\det A$（$i=1,\dots,n$）。

### 2.5 可逆判据

对 $n\times n$ 矩阵 $A$，下列条件等价：$\det A\neq0$；$A$ 可逆；$\operatorname{rank}(A)=n$（列、行线性无关）；零空间只有 $\mathbf{0}$（$A\mathbf{x}=\mathbf{b}$ 对任意 $\mathbf{b}$ 有唯一解）；$0$ 不是 $A$ 的特征值。

## 3. 为什么这样定义

### 3.1 $A\operatorname{adj}(A)=\det(A)I$

看第 $i$ 行与 $\operatorname{adj}(A)$ 第 $i$ 列的内积：$\sum_k a_{ik}\operatorname{adj}(A)_{ki}=\sum_k a_{ik}C_{ik}=\det A$，最后一步正是沿第 $i$ 行的余子式展开。

再看第 $i$ 行与第 $j$ 列（$i\neq j$）的内积：$\sum_k a_{ik}C_{jk}$。这等于把 $A$ 的第 $j$ 行换成第 $i$ 行后的行列式；此时两行相同，值为 0。于是

$$
A\operatorname{adj}(A)=\det(A)I_n.
$$

同理 $\operatorname{adj}(A)A=\det(A)I_n$。两边除以 $\det A$ 即得逆矩阵公式。

### 3.2 Cramer 法则从哪来

由 $\mathbf{x}=A^{-1}\mathbf{b}=\operatorname{adj}(A)\mathbf{b}/\det A$，第 $i$ 个分量为 $x_i=\frac{1}{\det A}\sum_j\operatorname{adj}(A)_{ij}b_j=\frac{1}{\det A}\sum_j C_{ji}b_j$。右边分子正好是 $A_i$ 沿第 $i$ 列的展开：$A_i$ 的第 $i$ 列是 $\mathbf{b}$，其余列与 $A$ 相同。这就是 Cramer 法则。

## 4. 性质与推导

### 4.1 常用恒等式

- $\operatorname{adj}(AB)=\operatorname{adj}(B)\operatorname{adj}(A)$；
- $\det(\operatorname{adj}A)=(\det A)^{n-1}$；
- $\operatorname{adj}(\operatorname{adj}A)=(\det A)^{n-2}A$（$n\ge2$）；
- 对 $2\times2$：$\operatorname{adj}\begin{bmatrix}a&b\\c&d\end{bmatrix}=\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$。

### 4.2 复杂度与数值稳定性

Cramer 法则要算 $n+1$ 个行列式。每个若按 Leibniz 是 $O(n!)$；即使每个都用消元，也是 $O(n^4)$，而 LU 解方程只要 $O(n^3)$。

数值上更糟：行列式可能极大或极小，计算中有大量相消；解的相对误差大致被条件数（condition number）$\operatorname{cond}(A)$ 放大。实际软件用带主元的 LU（或 QR、SVD），Cramer 只作为理论工具和 $2\times2$、$3\times3$ 的闭式公式。

### 4.3 为什么理论上有价值

Cramer 明确显示：每个 $x_i$ 都是 $\mathbf{b}$ 的**线性函数**，系数是 $\det A_i/\det A$。做灵敏度分析、证明解的唯一性、推导伴随矩阵性质时，这种显式表达很方便。

## 5. 例子

### 5.1 手算：$2\times2$ 的逆与 Cramer

$$
A=\begin{bmatrix}3&1\\1&2\end{bmatrix},\quad \det A=5,\quad
\operatorname{adj}(A)=\begin{bmatrix}2&-1\\-1&3\end{bmatrix},\quad
A^{-1}=\frac15\begin{bmatrix}2&-1\\-1&3\end{bmatrix}.
$$

验算 $AA^{-1}=I$：第一行乘第一列 $(6-1)/5=1$，第一行乘第二列 $(-3+3)/5=0$，其余同理。

解 $A\mathbf{x}=(5,5)^\top$。Cramer：$\det A_1=\begin{vmatrix}5&1\\5&2\end{vmatrix}=5$，所以 $x=1$；$\det A_2=\begin{vmatrix}3&5\\1&5\end{vmatrix}=10$，所以 $y=2$。与消元结果一致。

### 5.2 手算：$3\times3$ 的伴随与逆

$$
A=\begin{bmatrix}1&2&0\\0&1&1\\1&0&1\end{bmatrix},\qquad
\det A=1(1\cdot1-1\cdot0)-2(0\cdot1-1\cdot1)+0=3.
$$

九个代数余子式排成

$$
C=\begin{bmatrix}1&1&-1\\-2&1&2\\2&-1&1\end{bmatrix},\qquad
\operatorname{adj}(A)=C^\top=\begin{bmatrix}1&-2&2\\1&1&-1\\-1&2&1\end{bmatrix}.
$$

于是 $A^{-1}=\frac13\begin{bmatrix}1&-2&2\\1&1&-1\\-1&2&1\end{bmatrix}$。验算 $A\operatorname{adj}(A)=3I$：例如第一行乘第一列 $1+2+0=3$，第一行乘第二列 $-2+2+0=0$。

### 5.3 手算：$3\times3$ 的 Cramer

仍用上面的 $A$，取 $\mathbf{b}=(1,0,0)^\top$。把 $\mathbf{b}$ 换到第 1 列：

$$
A_1=\begin{bmatrix}1&2&0\\0&1&1\\0&0&1\end{bmatrix},\quad \det A_1=1.
$$

换到第 2 列得 $\det A_2=1$，换到第 3 列得 $\det A_3=-1$。因此

$$
\mathbf{x}=\frac13(1,\,1,\,-1)^\top.
$$

代回验证：第一行 $1/3+2/3=1$；第二行 $1/3-1/3=0$；第三行 $1/3-1/3=0$。

## 6. 常见误区

1. **误区**：$\operatorname{adj}(A)$ 就是代数余子式矩阵 $C$。
   **纠正**：还要转置，$\operatorname{adj}(A)=C^\top$。用 $2\times2$ 检查：$C=\begin{bmatrix}d&-c\\-b&a\end{bmatrix}$，转置后才得到 $\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$。

2. **误区**：$\det A$ 很小就说明 $A$ 接近奇异、逆很大。
   **纠正**：$0.1I_{100}$ 的 $\det=10^{-100}$，但逆是 $10I_{100}$，条件数为 1。尺度会改变 $\det$，不改变条件数。

3. **误区**：Cramer 法则对任何线性方程组都能用。
   **纠正**：只适用于方阵且 $\det A\neq0$。超定、欠定或奇异系统要用最小二乘、伪逆或消元。

4. **误区**：逆矩阵的元素就是余子式除以 $\det A$。
   **纠正**：位置是转置的。写错转置是 $3\times3$ 手算最常见的错误；用 $A\operatorname{adj}(A)=\det(A)I$ 验算最稳。

## 7. 概念网络

- 前置：[5.1 行列式几何](01-行列式几何.md) —— $\det$ 是体积缩放，$\det=0$ 表示降维。
- 前置：[5.2 行列式的性质与计算](02-行列式的性质与计算.md) —— 余子式展开是伴随矩阵的零件。
- 后续：[5.4 行列式与 Jacobian](04-行列式与Jacobian.md) —— 逆映射的 Jacobian 是 $J^{-1}$。
- 关联：[3.4 逆矩阵](../03-矩阵与线性映射/04-逆矩阵与可逆性.md) —— 逆的消元算法与这里的分式公式互为验证。
- 关联：[4.4 最小二乘](../04-线性方程组/04-最小二乘.md) —— 不可逆时用伪逆代替逆。
- 关联：[4.6 条件数](../04-线性方程组/06-条件数与病态.md) —— $\det$ 小不等于病态，判稳定性看条件数。
- 关联：9.2 多元高斯—— 协方差逆与 $\log\det$ 出现在密度里。

## 8. 通往大模型：高斯密度与流模型里的「逆」

**出现位置**：多元高斯（multivariate Gaussian）的对数密度、归一化流的似然、以及任何要「解一个线性系统」的层。

**形状**：协方差 $\Sigma\in\mathbb{R}^{D\times D}$，$D$ 可能是 512 或 4096；精度矩阵 $\Sigma^{-1}$ 同样大小。

**公式**：

$$
\log p(\mathbf{x})=-\frac{D}{2}\log(2\pi)-\frac12\log\det\Sigma
-\frac12(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu).
$$

**为什么不用 Cramer**：$D=4096$ 时，Cramer 要 4097 个 $4096\times4096$ 行列式，复杂度 $O(D^4)$，且数值极不稳定。实际用 Cholesky 分解 $\Sigma=LL^\top$：$\log\det\Sigma=2\sum_i\log L_{ii}$，解 $L\mathbf{z}=\mathbf{x}-\boldsymbol\mu$ 后算 $\|\mathbf{z}\|^2$，全程 $O(D^3)$ 且不用显式求逆。

**可检验结论**：取 $\Sigma=\operatorname{diag}(4,9)$，$\log\det\Sigma=\log36\approx3.584$，在均值处的密度是 $1/(2\pi\cdot6)=1/(12\pi)\approx0.0265$。若把 $\Sigma$ 换成 $0.01\Sigma$，$\det$ 缩小 $10^{-4}$，密度放大 $10^4$——这正是「$\det$ 依赖尺度」的直观版本。

## 9. 练习

1. **概念**：为什么 $\operatorname{adj}(A)$ 要转置？用 $2\times2$ 的 $C$ 与 $\operatorname{adj}(A)$ 说明。
2. **计算**：$A=\begin{bmatrix}2&1\\5&3\end{bmatrix}$。求 $\det A$、$A^{-1}$，并解 $A\mathbf{x}=(4,7)^\top$。
3. **计算**：$B=\begin{bmatrix}2&1&0\\1&2&1\\0&1&2\end{bmatrix}$。求 $\det B$、$\operatorname{adj}(B)$、$B^{-1}$，并验证 $B\operatorname{adj}(B)=4I$。
4. **计算**：用 Cramer 法则解 $B\mathbf{x}=(1,2,3)^\top$（$B$ 同上）。
5. **开放**：既然 Cramer 法则有闭式解，为什么数值库不实现它？从复杂度、稳定性、适用范围三个角度回答。

<details>
<summary>参考答案</summary>

1. 因为逆矩阵第 $i$ 行第 $j$ 列要用「删掉第 $j$ 行第 $i$ 列」的余子式，索引自然转置。$2\times2$ 中 $C=\begin{bmatrix}d&-c\\-b&a\end{bmatrix}$，而 $\operatorname{adj}(A)=\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$；不转置的话 $A C\neq(\det A)I$，而 $A C^\top=(\det A)I$ 才成立。
2. $\det=6-5=1$；$A^{-1}=\begin{bmatrix}3&-1\\-5&2\end{bmatrix}$；$\mathbf{x}=A^{-1}(4,7)^\top=(12-7,\,-20+14)^\top=(5,-6)^\top$。验证 $A(5,-6)=(4,7)$。
3. $\det B=2(4-1)-1(2-0)=4$；$\operatorname{adj}(B)=\begin{bmatrix}3&-2&1\\-2&4&-2\\1&-2&3\end{bmatrix}$；$B^{-1}=\frac14\operatorname{adj}(B)$；$B\operatorname{adj}(B)=4I$。
4. $\det B_1=2$，$\det B_2=0$，$\det B_3=6$，所以 $\mathbf{x}=(2,0,6)/4=(1/2,\,0,\,3/2)^\top$。验证：$B\mathbf{x}=(1,2,3)^\top$。
5. 复杂度：$n+1$ 个行列式，每个 $O(n^3)$，共 $O(n^4)$，而 LU 是 $O(n^3)$。稳定性：行列式容易上溢/下溢且相消严重，误差被 $\operatorname{cond}(A)$ 放大。适用范围：只对方阵且 $\det\neq0$ 有效；超定/欠定要用最小二乘或伪逆。它适合教学与 $2\times2$、$3\times3$ 闭式推导。

</details>

## 10. 一页速记

- $C_{ij}=(-1)^{i+j}M_{ij}$；$\operatorname{adj}(A)=C^\top$，下标是 $ji$。
- $A^{-1}=\operatorname{adj}(A)/\det A$，且 $A\operatorname{adj}(A)=\det(A)I$。
- Cramer：$x_i=\det(A_i)/\det A$，$A_i$ 把第 $i$ 列换成 $\mathbf{b}$。
- 可逆 $\iff\det\neq0\iff$ 满秩 $\iff$ 零空间为 $\{\mathbf{0}\}$。
- Cramer 是 $O(n^4)$ 且数值不稳；实际用 LU/QR/SVD。

---

下一篇 → [5.4 行列式与 Jacobian](04-行列式与Jacobian.md)
