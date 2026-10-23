---
id: 7.2
title: SVD 的计算与性质
module: M07
slug: svd-computation
prereq: [6.4, 7.1]
minutes: 40
level: 核心
tags: [奇异值, A转置A, 谱范数, 条件数, 数值秩]
llm_hooks: [numerical-svd, spectral-norm, condition-number]
---

# 7.2 SVD 的计算与性质

> **本节位置**：模块 7 · SVD ｜ **前置**：[7.1](01-SVD的几何.md) ｜ **预计**：40 分钟
>
> **一句话**：奇异值就是 $A^\top A$ 特征值的平方根，$A^\top A$ 的特征向量就是右奇异向量——但真正算的时候别真的去乘 $A^\top A$。

## 0. 场景引入

你要对一个 $3000\times2000$ 的矩阵做 SVD。教科书给的办法是：

1. 算 $A^\top A$（$2000\times2000$）；
2. 对它做特征分解，特征值开方得 $\sigma_i$，再由 $\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$ 得左奇异向量。

按这个流程，你用双精度算出最小的奇异值是 $3.2\times10^{-9}$，于是判定矩阵秩亏。换用 LAPACK 的 `svd` 直接算，同一个矩阵的最小奇异值是 $1.1\times10^{-6}$——差了约 350 倍。谁错了？问题出在第一步：$A^\top A$ 把条件数（condition number）**平方**了。这一节讲清三件事：SVD 与特征分解的精确关系、由它导出的性质清单、以及为什么实际实现不这么算。

## 1. 直觉

$A^\top A$ 是「从输入侧看 $A$」：它记录每个输入方向被放大的**平方**。因为 $\|A\mathbf{v}\|^2=\mathbf{v}^\top A^\top A\mathbf{v}$，若 $\mathbf{v}$ 是 $A^\top A$ 的特征向量、特征值为 $\lambda$，则 $\|A\mathbf{v}\|^2=\lambda$，拉伸倍数是 $\sqrt\lambda$——这就是奇异值。所以 $A^\top A$ 的特征向量 = 右奇异向量，特征值 = $\sigma^2$。

对称地，$AA^\top$ 是「从输出侧看 $A$」，它的特征向量是左奇异向量，非零特征值同样是 $\sigma^2$。两张「X 光片」看的是同一副骨架。

> 💡 **一句话直觉**：**SVD = 对 $A^\top A$ 做谱分解，再开方；$U$ 由 $A$ 把 $V$ 搬过去得到。**

类比边界：这个「X 光片」要付出代价——$\lambda=\sigma^2$ 把动态范围平方。$\sigma$ 从 $10^{0}$ 到 $10^{-8}$ 时，$\lambda$ 从 $10^{0}$ 到 $10^{-16}$，正好撞上双精度的分辨率（$2^{-52}\approx2.2\times10^{-16}$）。图像还在，细节已经糊了。

## 2. 定义

### 2.1 由 $A^\top A$ 构造 SVD（存在性版本）

设 $A\in\mathbb{R}^{m\times n}$，$G=A^\top A$。则 $G$ 对称半正定，由谱定理存在正交 $V$ 使

$$
G=V\Lambda V^\top,\qquad \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n),\ \lambda_1\ge\cdots\ge\lambda_n\ge0
$$

流程：

1. 对 $G$ 做特征分解，得 $\lambda_i$ 与单位特征向量 $\mathbf{v}_i$；
2. 令 $\sigma_i=\sqrt{\lambda_i}$，设前 $r$ 个为正，则 $\operatorname{rank}(A)=r$；
3. 对 $i\le r$ 定义 $\mathbf{u}_i=A\mathbf{v}_i/\sigma_i\in\mathbb{R}^m$；
4. $\mathbf{u}_1,\dots,\mathbf{u}_r$ 已标准正交（7.1 的 4.3），再补 $m-r$ 个单位向量使 $U$ 正交（它们张成左零空间）；
5. 取 $\Sigma\in\mathbb{R}^{m\times n}$，对角元为 $\sigma_1,\dots,\sigma_r$，其余为 0。

得到 $A=U\Sigma V^\top$。注意 $V$ 是 $n\times n$、$U$ 是 $m\times m$，而 $\Sigma$ 与 $A$ **同形**。

### 2.2 三个矩阵的关系

| 矩阵 | 形状 | 特征值 | 特征向量 |
|---|---|---|---|
| $A^\top A$ | $n\times n$ | $\sigma_i^2$（含 $n-r$ 个 0） | 右奇异向量 $\mathbf{v}_i$ |
| $AA^\top$ | $m\times m$ | $\sigma_i^2$（含 $m-r$ 个 0） | 左奇异向量 $\mathbf{u}_i$ |
| $A$ | $m\times n$ | 仅当 $m=n$ 才有定义 | 与奇异向量一般不同 |

$A^\top A$ 与 $AA^\top$ 的非零特征值**完全相同**，都等于 $\sigma_i^2$。差别只在零特征值的个数：分别是 $n-r$ 和 $m-r$。

### 2.3 实际算法：不要显式构造 $A^\top A$

LAPACK 的 `gesdd`/`gesvd` 走的是**双对角化 + 隐式 QR**：先用 Householder 变换把 $A$ 化成上双对角矩阵（$O(mn^2)$），再迭代求奇异值。全程只对 $A$ 做正交变换，从不形成 $A^\top A$。NumPy 的 `np.linalg.svd`、PyTorch 的 `torch.linalg.svd` 都调用它。

## 3. 为什么这样定义

**为什么 $A^\top A$ 是半正定的？** 对任意 $\mathbf{x}$ 有 $\mathbf{x}^\top A^\top A\mathbf{x}=\|A\mathbf{x}\|_2^2\ge0$，二次型永不为负，所以特征值 $\lambda_i\ge0$，开方总合法。这是奇异值非负的代数根源。

**为什么用 $A^\top A$ 而不是随便一个对称矩阵？** 因为它恰好编码了「所有输入方向被拉伸的平方」，而谱定理保证它有完整的正交特征基。$A$ 本身可能没有实特征值（如旋转矩阵），但 $A^\top A$ 一定有，而且是对称的——这就是 SVD 比特征分解普适的原因。

**为什么还要补 $U$ 的剩余列？** 当 $m>r$ 时，$A\mathbf{v}_i/\sigma_i$ 只给出 $r$ 个方向，$U$ 需要 $m$ 个。缺的那些对应「任何输入都到不了的方向」，即左零空间，可以任意选一组标准正交基，它们不影响 $A=U\Sigma V^\top$（因为乘的 $\sigma$ 是 0）。

## 4. 性质与推导

### 4.1 核心推导：$\|A\|_F^2=\sum_i\sigma_i^2$

$$
\|A\|_F^2=\operatorname{tr}(A^\top A)=\operatorname{tr}(V\Lambda V^\top)=\operatorname{tr}(\Lambda)=\sum_i\lambda_i=\sum_i\sigma_i^2
$$

第二、三步用到「迹在正交相似下不变」。这个式子说的是：**Frobenius 范数把所有奇异值平方加起来**，因此低秩近似的误差可以用被丢掉的那部分奇异值精确计算（第 7.3 节）。

### 4.2 性质清单

| 性质 | 式子 | 说明 |
|---|---|---|
| 谱范数（spectral norm） | $\|A\|_2=\sigma_1$ | 最大拉伸倍数 |
| 最小奇异值 | $\sigma_{\min}=1/\|A^{+}\|_2$ | 可逆时 $\sigma_n=1/\|A^{-1}\|$ |
| 条件数 | $\kappa_2(A)=\sigma_1/\sigma_r$ | 误差放大上界；见 4.6 |
| 秩 | $\operatorname{rank}(A)=\#\{\sigma_i>0\}$ | 数值上用阈值判 |
| Frobenius | $\|A\|_F^2=\sum\sigma_i^2$ | 见 4.1 |
| 扰动稳定 | $\lvert\sigma_i(A+E)-\sigma_i(A)\rvert\le\|E\|_2$ | Weyl 不等式：奇异值对扰动不敏感 |

最后一条值得强调：**奇异值稳定，奇异向量不一定**。$\sigma_i$ 与 $\sigma_{i+1}$ 很接近时，对应的两个方向可以剧烈旋转；$\sigma_i$ 与 $\sigma_{i+1}$ 相距很远时，$\mathbf{u}_i,\mathbf{v}_i$ 才稳定。

### 4.3 与特征分解的区别

| 维度 | 特征分解 | SVD |
|---|---|---|
| 存在条件 | 方阵且可对角化 | 任意 $m\times n$ |
| 基 | 一组（$P$，一般不正交） | 两组（$U,V$，都正交） |
| 取值 | 可为负、可为复数 | 一定非负实数 |
| 与转置 | $A,A^\top$ 特征值相同 | 奇异值相同，向量互换 |
| 关系 | 正规矩阵：$\sigma_i=\lvert\lambda_i\rvert$ | 一般矩阵：$\sigma_i\neq\lvert\lambda_i\rvert$ |
| 扰动 | 特征值可能剧变（非正规矩阵） | $\lvert\Delta\sigma\rvert\le\|E\|_2$ |

正规矩阵（normal matrix）指 $A^\top A=AA^\top$，包含对称、反对称、正交矩阵。只有在这类矩阵上，SVD 与特征分解才「几乎同一件事」。

## 5. 例子

### 5.1 手算：$3\times2$ 矩阵的完整 SVD

$$
A=\begin{bmatrix}1&1\\1&0\\0&1\end{bmatrix},\qquad
A^\top A=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

**特征值**：$\det(G-\lambda I)=(2-\lambda)^2-1=0\Rightarrow\lambda_1=3,\lambda_2=1$。

**右奇异向量**：$\lambda_1=3$ 时 $-x+y=0$，取 $\mathbf{v}_1=\frac{1}{\sqrt2}(1,1)^\top$；$\lambda_2=1$ 时 $x+y=0$，取 $\mathbf{v}_2=\frac{1}{\sqrt2}(1,-1)^\top$。

**奇异值**：$\sigma_1=\sqrt3\approx1.732$，$\sigma_2=1$。

**左奇异向量**：

$$
\mathbf{u}_1=\frac{A\mathbf{v}_1}{\sqrt3}=\frac{(2,1,1)/\sqrt2}{\sqrt3}=\frac{1}{\sqrt6}\begin{bmatrix}2\\1\\1\end{bmatrix},
\qquad
\mathbf{u}_2=A\mathbf{v}_2=\frac{1}{\sqrt2}\begin{bmatrix}0\\1\\-1\end{bmatrix}
$$

只有 2 个，还需第 3 个。取 $\mathbf{u}_3=\mathbf{u}_1\times\mathbf{u}_2=\frac{1}{\sqrt3}(-1,1,1)^\top$，验证：$\|\mathbf{u}_3\|=1$，$\mathbf{u}_3^\top\mathbf{u}_1=(-2+1+1)/\sqrt{18}=0$ ✓。

**核对**：$\|A\|_F^2=4$，而 $\sigma_1^2+\sigma_2^2=3+1=4$ ✓；$\kappa_2=\sqrt3/1\approx1.732$；$A$ 不是方阵，**根本不存在特征值**——这正是 SVD 的用武之地。

### 5.2 手算：为什么别乘 $A^\top A$

考虑 $A=\begin{bmatrix}1&1\\1&1+\varepsilon\end{bmatrix}$，$\varepsilon=10^{-8}$。理论上

$$
A^\top A=\begin{bmatrix}2&2+\varepsilon\\2+\varepsilon&2+2\varepsilon+\varepsilon^2\end{bmatrix},\qquad
\det(A^\top A)=\varepsilon^2
$$

于是 $\lambda_2\approx\varepsilon^2/4=2.5\times10^{-17}$ 量级，$\sigma_2\approx\varepsilon/2=5\times10^{-9}$，$\kappa_2\approx4\times10^{8}$。双精度只有约 16 位十进制有效数字：直接对 $A^\top A$ 做特征分解，$\lambda_2$ 会被舍入误差淹没（实测常直接算成 0），符号甚至可能变负——开方就得到 `nan` 或虚假的秩。

用 `np.linalg.svd(A)` 直接算则稳定得多，因为算法从不对 $A$ 平方。**结论**：手算和证明用 $A^\top A$，数值计算用 LAPACK 的 SVD。

## 6. 常见误区

1. **误区**：奇异值是「$A$ 的特征值开平方」。
   **纠正**：是 $A^\top A$（或 $AA^\top$）特征值的平方根。$A$ 可能没有实特征值，$\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ 的特征值是 $\pm i$，奇异值却是 $1,1$。
2. **误区**：$\sigma_i=\lvert\lambda_i\rvert$ 总是成立。
   **纠正**：只对正规矩阵成立。$\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 特征值双 1，奇异值是 $1.618$ 与 $0.618$（黄金比例）。
3. **误区**：算出一个很小的奇异值就说明矩阵秩亏。
   **纠正**：先看阈值。NumPy 的 `matrix_rank` 用 $\sigma_i>\max(m,n)\cdot\varepsilon\cdot\sigma_1$；低于阈值的才算 0（数值秩，第 7.4 节）。
4. **误区**：奇异值算得准，奇异向量也算得准。
   **纠正**：$\sigma_i\approx\sigma_{i+1}$ 时，对应子空间内的任意正交组合都是合法的奇异向量，单个向量的数值结果没有意义，只有**子空间**稳定。

## 7. 概念网络

- 前置：[6.4 对称矩阵与谱定理](../06-特征值/04-对称矩阵与谱定理.md) —— 对称矩阵有正交特征基，SVD 由此构造。
- 前置：[7.1 SVD 的几何](01-SVD的几何.md) —— $U,V$ 的几何含义。
- 前置：[2.6 秩](../02-向量空间/06-秩.md) —— 秩 = 非零奇异值个数。
- 后续：[7.3 低秩近似](03-低秩近似.md) —— 用 $\sum_{i>k}\sigma_i^2$ 写误差。
- 后续：[7.4 伪逆与数值秩](04-伪逆与数值秩.md) —— 阈值、$\kappa$、截断。
- 关联：[4.6 条件数与病态](../04-线性方程组/06-条件数与病态.md) —— $\kappa=\sigma_1/\sigma_r$ 的误差放大解释。

## 8. 通往大模型：算得对，才算得动

**出现位置**：分析权重谱、估计谱范数做谱归一化、诊断微调是否病态、以及在低精度下判断「哪些奇异值还可信」。

**形状与代价**：对 $W\in\mathbb{R}^{4096\times4096}$ 做完整 SVD 约需 $O(D^3)\approx6.9\times10^{10}$ 次浮点运算，显存里还要放 $4096^2$ 的 $U$ 和 $V$：$3\times4096^2\times4$ 字节（fp32）$\approx201$ MB。实际分析常常只求前 $k$ 个（`svds`，$O(kD^2)$）。

**为什么需要**：$\|W\|_2=\sigma_1$ 是谱归一化（spectral normalization）要约束的量——把 $\sigma_1$ 压到 1 附近，判别器的 Lipschitz 常数就有上界。训练中还可以用幂迭代估计 $\sigma_1$：迭代 $\mathbf{x}\leftarrow W^\top W\mathbf{x}$ 再归一化，10–20 步就够，代价只有两次矩阵向量乘。

**低精度陷阱**：bf16 只有 8 位有效位（1 位隐含 + 7 位显式），相对精度约 $2^{-8}\approx4\times10^{-3}$。$\sigma_{\min}/\sigma_1$ 低于 $10^{-2}$ 量级时，尾部奇异值在 bf16 下已不可分辨；fp32 相对精度约 $6\times10^{-8}$，门槛降到 $10^{-7}$ 量级。一个 $\kappa\approx10^{6}$ 的 $4096\times4096$ 权重在 bf16 下，最小的那些奇异值只是噪声，不能用来解释「模型学到了什么」。

**可检验结论**：对随机 $W$（元素方差 $1/D$），奇异值谱落在 $[0,2]$ 附近（Marchenko–Pastur 律），$\sigma_1\approx2$、$\sigma_{\min}$ 接近 0，$\kappa$ 随 $D$ 增大而不是 1；因此**未训练的权重天然病态**，微调时看到的大 $\kappa$ 未必是训练造成的。

```python
s = np.linalg.svd(W, compute_uv=False)   # W: [4096, 4096]
print(s[0], s[-1], s[0] / s[-1])
```

## 9. 练习

1. **概念**：为什么 $A^\top A$ 的特征值一定非负？用一行式子说明。
2. **计算**：$A=\begin{bmatrix}3&0\\4&5\end{bmatrix}$ 由 $A^\top A$ 求全部奇异值并验证 $\lvert\det A\rvert=\sigma_1\sigma_2$；再对 $B=\begin{bmatrix}2&0\\0&0\end{bmatrix}$ 求 $\sigma_1,\sigma_2$、$\|B\|_2$、$\|B\|_F$、$\kappa_2$，$\kappa$ 是有限值吗？
3. **场景**：某权重矩阵的奇异值为 $(2.0,1.9,0.01,\dots)$。它的前两个奇异向量在你的数值实验里两次运行结果不同，合理吗？
4. **开放**：既然奇异值稳定而奇异向量不稳定，做「权重可解释性分析」时应该报告哪一个？

<details>
<summary>参考答案</summary>

1. $\mathbf{x}^\top A^\top A\mathbf{x}=\|A\mathbf{x}\|_2^2\ge0$，二次型非负 $\Rightarrow$ 特征值非负。
2. $A^\top A=\begin{bmatrix}25&20\\20&25\end{bmatrix}$，$\lambda=45,5$，$\sigma=(3\sqrt5,\sqrt5)$，$\sigma_1\sigma_2=15=\lvert\det A\rvert$ ✓。对 $B$：$\sigma=(2,0)$，$\|B\|_2=\|B\|_F=2$，$\kappa_2=\infty$（奇异）。
3. 合理。$2.0$ 与 $1.9$ 很接近（相对差 5%），对应二维子空间内的方向可以任意旋转；稳定的是**子空间**（张成的平面），不是单个向量。
4. 优先报告奇异值谱与低维子空间（如「前 10 维解释了 90% 能量」），少报单个奇异向量的坐标；若必须报向量，给出子空间距离与 $\sigma_i/\sigma_{i+1}$ 作为置信度。

</details>

## 10. 一页速记

- $\sigma_i=\sqrt{\lambda_i(A^\top A)}$；$V$ 是 $A^\top A$ 的特征向量，$\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$。
- $A^\top A$ 与 $AA^\top$ 非零特征值相同，都是 $\sigma_i^2$。
- $\|A\|_2=\sigma_1$，$\|A\|_F^2=\sum\sigma_i^2$，$\kappa_2=\sigma_1/\sigma_r$，$\lvert\det A\rvert=\prod\sigma_i$。
- 特征分解只对方阵、可能有复数；SVD 对任意矩阵、一定非负；手算用 $A^\top A$，数值计算用 LAPACK，显式乘 $A^\top A$ 会把 $\kappa$ 平方。

---

下一篇 → [7.3 低秩近似与 Eckart–Young](03-低秩近似.md)
