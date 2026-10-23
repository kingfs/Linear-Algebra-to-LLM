---
id: 6.7
title: Jordan 标准形与矩阵函数
module: M06
slug: jordan-form
prereq: [6.3]
minutes: 35
level: 进阶
tags: [Jordan 标准形, 亏损矩阵, 矩阵指数, 线性 ODE]
llm_hooks: [matrix-exponential, stability, transient-growth]
---

# 6.7 Jordan 标准形与矩阵函数

> **本节位置**：模块 6 · 特征值 ｜ **前置**：[6.3 对角化](03-对角化.md) ｜ **预计**：35 分钟
>
> **一句话**：不能对角化的矩阵仍可化成 Jordan 标准形；对角线上是特征值，上对角线上是 1，这些 1 让矩阵幂和 $e^{At}$ 多出 $t,t^2/2,\dots$ 这样的多项式因子。

## 0. 场景引入

线性微分方程组 $\dot{\mathbf{x}}=A\mathbf{x}$ 的解是 $\mathbf{x}(t)=e^{At}\mathbf{x}(0)$。若 $A$ 可对角化，$e^{At}=Pe^{\Lambda t}P^{-1}$，每个分量是 $e^{\lambda_i t}$ 的叠加。

但 $A=\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 只有一个特征方向，$P$ 不可逆，这条路走不通。**问题**：$e^{At}$ 还能算吗？解会变成什么样？重复特征值会不会让解「长出一个 $t$」？

## 1. 直觉

对角化矩阵是「纯伸缩」；不能对角化的矩阵在伸缩之外还有「剪切」。最简单的剪切是 $N=\begin{bmatrix}0&1\\0&0\end{bmatrix}$：它把 $\mathbf{e}_2$ 送到 $\mathbf{e}_1$，把 $\mathbf{e}_1$ 送到零。$N^2=0$——剪切两次就没了。这种「几次之后归零」的矩阵叫**幂零矩阵（nilpotent matrix）**。

把 $A=\lambda I+N$ 拆成「伸缩 + 剪切」。因为 $\lambda I$ 与 $N$ 可交换，二项式定理成立：

$$
(\lambda I+N)^k=\sum_{j=0}^{k}\binom{k}{j}\lambda^{k-j}N^j
$$

而 $N^2=0$，所以只剩前两项：$A^k=\lambda^kI+k\lambda^{k-1}N$。那个 $k$ 就是多项式因子的来源。

> 💡 **一句话直觉**：**Jordan 块 = 特征值在对角线上，剪切在上对角线上；剪切是幂零的，所以矩阵函数只多出有限项多项式。**

类比边界：Jordan 块的大小等于剪切链的长度；链越长，多项式因子次数越高。可对角化时所有块都是 $1\times1$，没有剪切。

## 2. 定义

### 2.1 Jordan 块

**Jordan 块（Jordan block）** $J_k(\lambda)$ 是 $k\times k$ 矩阵：

$$
J_k(\lambda)=\begin{bmatrix}
\lambda&1&&\\
&\lambda&\ddots&\\
&&\ddots&1\\
&&&\lambda
\end{bmatrix}=\lambda I_k+N_k
$$

其中 $N_k$ 的上对角线全为 1、其余为 0，满足 $N_k^k=0$、$N_k^{k-1}\neq0$。

### 2.2 Jordan 标准形

**Jordan 标准形（Jordan canonical form）**：任意复矩阵 $A$ 都相似于分块对角的 $J=\operatorname{diag}(J_{k_1}(\lambda_1),\dots,J_{k_m}(\lambda_m))$，即存在可逆 $P$ 使 $A=PJP^{-1}$；不计块顺序，$J$ 唯一。

对每个特征值 $\lambda$：代数重数等于该 $\lambda$ 的所有块大小之和；几何重数等于块数；最大块的大小是最小多项式（使 $p(A)=0$ 的最低次多项式）中 $(\lambda-\lambda_0)$ 的重数。所以**可对角化 $\Leftrightarrow$ 所有 Jordan 块都是 $1\times1$**，即几何重数等于代数重数。

### 2.3 广义特征向量

若 $(A-\lambda I)^m\mathbf{v}=\mathbf{0}$ 但 $(A-\lambda I)^{m-1}\mathbf{v}\neq\mathbf{0}$，称 $\mathbf{v}$ 是秩 $m$ 的**广义特征向量（generalized eigenvector）**。它们组成 **Jordan 链（Jordan chain）**：

$$
(A-\lambda I)\mathbf{v}_m=\mathbf{v}_{m-1},\quad\dots,\quad(A-\lambda I)\mathbf{v}_2=\mathbf{v}_1,\quad(A-\lambda I)\mathbf{v}_1=\mathbf{0}
$$

其中 $\mathbf{v}_1$ 是普通特征向量。链长就是对应 Jordan 块的大小。

### 2.4 矩阵函数

对解析函数 $f$，矩阵函数 $f(A)$ 由幂级数定义；若 $A=PJP^{-1}$，则 $f(A)=Pf(J)P^{-1}$。对 $2\times2$ 的 Jordan 块，

$$
f(J_2(\lambda))=\begin{bmatrix}f(\lambda)&f'(\lambda)\\0&f(\lambda)\end{bmatrix}
$$

对角线上是 $f(\lambda)$，第 $m$ 条上对角线是 $f^{(m)}(\lambda)/m!$，一直排到 $m=k-1$。取 $f(x)=x^n$ 得 $f'(\lambda)=n\lambda^{n-1}$，正好是 §1 的 $A^k$ 公式。

## 3. 为什么这样定义

不是所有矩阵都能对角化：$\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 只有一条特征方向，旋转矩阵在实数域甚至没有实特征向量。若只研究可对角化矩阵，线性动力系统、矩阵幂、矩阵指数都会留下缺口。

Jordan 标准形补上这个缺口：它把所有矩阵按相似关系分类，每一类由一个分块对角的 $J$ 代表；两个矩阵相似，当且仅当它们的 Jordan 形（不计块顺序）相同。

但 Jordan 形在数值上很危险：它对扰动不连续。元素扰动 $10^{-8}$ 可能把一个大块拆成许多小块，$P$ 的条件数可以爆炸。所以实际计算用 **Schur 分解（Schur decomposition）** $A=QTQ^\top$（$Q$ 正交、$T$ 上三角），特征值在对角线上，数值稳定。Jordan 形是理论工具，不是计算工具。

## 4. 性质与推导

### 4.1 $2\times2$ 幂零块的幂

$A=\begin{bmatrix}1&1\\0&1\end{bmatrix}=I+N$，$N^2=0$，由二项式定理 $A^k=(I+N)^k=I+kN=\begin{bmatrix}1&k\\0&1\end{bmatrix}$。验证 $k=2$：$\begin{bmatrix}1&2\\0&1\end{bmatrix}$，与直接相乘一致。这不是纯缩放，而是多了一个随 $k$ 线性增长的项。

### 4.2 矩阵指数

**矩阵指数（matrix exponential）** 定义为 $e^{A}=\sum_{m=0}^{\infty}A^m/m!$。对 $A=\lambda I+N$（$N^k=0$），因为 $\lambda I$ 与 $N$ 可交换，

$$
e^{At}=e^{\lambda t}\sum_{m=0}^{k-1}\frac{(tN)^m}{m!}
=e^{\lambda t}\left(I+tN+\frac{t^2N^2}{2!}+\cdots+\frac{t^{k-1}N^{k-1}}{(k-1)!}\right)
$$

级数在有限项处截断，这是幂零性带来的最大好处。

### 4.3 线性 ODE 的解

对 $\dot{\mathbf{x}}=A\mathbf{x}$、$\mathbf{x}(0)=\mathbf{x}_0$，解是 $\mathbf{x}(t)=e^{At}\mathbf{x}_0$。对幂级数逐项求导：$\frac{d}{dt}e^{At}=Ae^{At}=e^{At}A$，所以 $\dot{\mathbf{x}}=A\mathbf{x}$，且 $t=0$ 时 $e^0=I$，初值吻合。$A$ 与 $e^{At}$ 可交换，因为 $e^{At}$ 是 $A$ 的幂级数。

### 4.4 稳定性与多项式因子

若 $\lambda=a+bi$，则 $|e^{\lambda t}|=e^{at}$。Jordan 块额外带来 $t^{m-1}$ 因子，但指数最终压倒多项式：$a<0$ 时 $t^{m-1}e^{at}\to0$。所以连续时间系统的稳定性只看 $\operatorname{Re}\lambda$：全部 $<0$ 渐近稳定，有 $>0$ 发散；$=0$ 时若块大小为 1 则有界振荡，块更大则出现多项式增长。离散时间看 $|\lambda|$：全部 $<1$ 稳定。实矩阵的复特征值 $a\pm bi$ 对应块 $\begin{bmatrix}a&-b\\b&a\end{bmatrix}$，其指数是 $e^{at}$ 乘旋转。

## 5. 例子

### 5.1 手算：$2\times2$ Jordan 块的幂与指数

取 $A=\begin{bmatrix}1&1\\0&1\end{bmatrix}=I+N$。幂：$A^k=\begin{bmatrix}1&k\\0&1\end{bmatrix}$。指数：

$$
e^{At}=e^{t}(I+tN)=e^{t}\begin{bmatrix}1&t\\0&1\end{bmatrix}
$$

取 $\mathbf{x}_0=(0,1)$，解为 $\mathbf{x}(t)=e^{t}(t,1)$。验证：$x_1'=e^t(t+1)=x_1+x_2$，$x_2'=e^t=x_2$，且 $x_1(0)=0$、$x_2(0)=1$。$t=1$ 时 $e^{A}=e\begin{bmatrix}1&1\\0&1\end{bmatrix}\approx\begin{bmatrix}2.718&2.718\\0&2.718\end{bmatrix}$。

### 5.2 $3\times3$ Jordan 块：二次多项式因子

取 $J=\begin{bmatrix}2&1&0\\0&2&1\\0&0&2\end{bmatrix}=2I+N$，$N^3=0$。则

$$
e^{Jt}=e^{2t}\begin{bmatrix}1&t&\frac{t^2}{2}\\0&1&t\\0&0&1\end{bmatrix}
$$

$t^2/2$ 来自 $N^2/2!$。若把 $2$ 换成 $-1$，则 $e^{-t}(1,t,t^2/2)$ 仍衰减到 0——多项式因子不改变最终的指数衰减，只改变衰减的细节。

### 5.3 场景：非正规矩阵的瞬态增长

取 $A=\begin{bmatrix}-1&100\\0&-1\end{bmatrix}=-I+100N$。特征值都是 $-1$，按特征值判据应当立刻衰减。但

$$
e^{At}=e^{-t}\begin{bmatrix}1&100t\\0&1\end{bmatrix}
$$

在 $t=0.01$ 时，矩阵约为 $\begin{bmatrix}0.990&0.990\\0&0.990\end{bmatrix}$，2-范数约 $1.60$，已超过初值 1；在 $t=1$ 时，右上角元素约 $36.8$，2-范数约 $36.8$，之后才指数衰减。峰值约为初值的 37 倍。这叫**瞬态增长（transient growth）**，是非正规矩阵（$AA^\top\neq A^\top A$）的典型现象。只看特征值会误判「没有放大」，实际扰动可能被放大数十倍。

## 6. 常见误区

1. **误区**：Jordan 标准形就是对角形。**纠正**：只有可对角化时才是；亏损矩阵的 Jordan 形上对角线有 1，不能继续对角化。
2. **误区**：特征值重复就一定不可对角化。**纠正**：$2I$ 的特征值 2 重数为 2，但已经是 Jordan 形（两个 $1\times1$ 块）；要看几何重数是否等于代数重数。
3. **误区**：Jordan 形唯一，所以 $P$ 也唯一。**纠正**：$J$ 在不计块顺序时唯一，$P$ 远不唯一；广义特征向量可以缩放、混合。
4. **误区**：$e^{A+B}=e^Ae^B$ 总成立。**纠正**：仅当 $AB=BA$ 时成立；$\lambda I$ 与 $N$ 可交换，所以 $e^{\lambda I+N}=e^{\lambda I}e^N$，一般的 $A,B$ 不行。
5. **误区**：实际计算应先算 Jordan 形再算 $e^{At}$。**纠正**：Jordan 形对扰动不连续，$P$ 可能极度病态；实际用 Schur 分解或 Padé 近似（`scipy.linalg.expm`）。
6. **误区**：特征值实部为负就一定单调衰减。**纠正**：非正规矩阵可能先瞬态增长（§5.3），Jordan 块还带来 $t^{m-1}$ 因子。

## 7. 概念网络

- 前置：[6.3 对角化](03-对角化.md) —— 可对角化是 Jordan 形的特例（所有块为 $1\times1$）。
- 前置：[6.2 特征多项式](02-特征多项式与迹.md) —— 代数重数、几何重数决定 Jordan 块的总大小与块数。
- 关联：[6.1 特征值与特征向量](01-特征值与特征向量.md) —— 广义特征向量是普通特征向量的链式延伸。
- 关联：[6.6 幂迭代](06-幂迭代与PageRank.md) —— 亏损矩阵的幂迭代会出现多项式因子，收敛比不再是纯几何级数。
- 后续：8.x 线性微分方程与矩阵指数 —— $\dot{\mathbf{x}}=A\mathbf{x}$ 的解就是 $e^{At}\mathbf{x}_0$（文档建设中）。

## 8. 通往大模型：连续时间动力学与瞬态增长

**出现位置**：神经 ODE、连续时间残差流、状态空间模型的连续化版本都写成 $\dot{\mathbf{h}}(t)=A\mathbf{h}(t)+B\mathbf{x}(t)$，解含 $e^{At}$，$A\in\mathbb{R}^{D\times D}$。训练中分析梯度传播时，也会遇到 $\prod_t W_t$ 这样的乘积，其局部行为由 Jacobian 的 Jordan 结构决定。

**为什么需要**：若 $A$ 可对角化，每个模式是纯指数 $e^{\lambda_i t}$；若 $A$ 亏损，会出现 $t^{m-1}e^{\lambda t}$，同样的特征值下衰减更慢、甚至先增长。§5.3 说明只看特征值不够：非正规矩阵可以在特征值全为负时仍瞬态放大扰动，这对「训练是否稳定」是实实在在的风险。矩阵指数还是线性 ODE、扩散模型连续时间采样、Neural ODE 数值积分器（如 `expm`）的核心。

**可检验结论**：对 $A=\begin{bmatrix}-1&100\\0&-1\end{bmatrix}$，$\|e^{At}\|_2$ 从 $t=0$ 的 1 升到 $t\approx1$ 的约 $36.8$，再衰减：先放大 37 倍，然后才「稳定」。用 `scipy.linalg.expm` 或幂级数截断求和都能复现；只看 `np.linalg.eigvals(A)` 会得到「全部 -1，稳定」的错误安心感。

## 9. 练习

1. **概念**：Jordan 块上对角线的 1 代表什么？为什么可对角化矩阵的 Jordan 形没有它？
2. **计算**：对 $A=\begin{bmatrix}2&1\\0&2\end{bmatrix}$，求 $A^k$ 与 $e^{At}$。
3. **计算**：对 $A=\begin{bmatrix}0&1\\-1&0\end{bmatrix}$，求 $e^{At}$，并说明它描述了什么运动。
4. **计算**：解 $\dot{\mathbf{x}}=A\mathbf{x}$，$A=\begin{bmatrix}-1&1\\0&-1\end{bmatrix}$，$\mathbf{x}(0)=(0,1)$，写出 $\mathbf{x}(t)$。
5. **开放**：为什么数值软件用 Schur 分解而不是 Jordan 形计算矩阵函数？举一个扰动改变 Jordan 结构的例子。

<details>
<summary>参考答案</summary>

1. 1 代表剪切：它把广义特征向量送到链上的前一个向量，使矩阵不能在该特征空间内被对角化。可对角化矩阵的每个特征空间都有完整特征向量，链长都是 1，所以没有上对角线的 1。
2. $A=2I+N$，$N^2=0$，故 $A^k=2^kI+k2^{k-1}N=\begin{bmatrix}2^k&k2^{k-1}\\0&2^k\end{bmatrix}$；$e^{At}=e^{2t}\begin{bmatrix}1&t\\0&1\end{bmatrix}$。
3. $A$ 是 $90^\circ$ 旋转的生成元，$e^{At}=\begin{bmatrix}\cos t&\sin t\\-\sin t&\cos t\end{bmatrix}$，描述单位圆上的匀速旋转，周期 $2\pi$。
4. $e^{At}=e^{-t}\begin{bmatrix}1&t\\0&1\end{bmatrix}$，故 $\mathbf{x}(t)=e^{-t}(t,1)$；验证 $x_1'=e^{-t}(1-t)=-x_1+x_2$，$x_2'=-e^{-t}=-x_2$。
5. Jordan 形对扰动不连续：$\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 扰动为 $\begin{bmatrix}1&\varepsilon\\0&1\end{bmatrix}$ 后，广义特征向量需要除以 $\varepsilon$，$P$ 的条件数趋于无穷，数值误差会把一个大块拆成小块。Schur 分解只用正交变换，条件数为 1，稳定可靠。

</details>

## 10. 一页速记

- Jordan 块 $J_k(\lambda)=\lambda I+N$，$N^k=0$，上对角线为 1。
- 任意复矩阵 $A=PJP^{-1}$；块的总大小 = 代数重数，块数 = 几何重数。
- 可对角化 $\Leftrightarrow$ 所有 Jordan 块为 $1\times1$。
- $f(J)$ 的对角线是 $f(\lambda)$，上对角线是 $f'(\lambda)$；$e^{At}=e^{\lambda t}\sum (tN)^m/m!$。
- 稳定性看 $\operatorname{Re}\lambda$（连续）或 $|\lambda|$（离散）；非正规矩阵可能有瞬态增长。

---

返回模块导览 → [M06 特征值](README.md)
