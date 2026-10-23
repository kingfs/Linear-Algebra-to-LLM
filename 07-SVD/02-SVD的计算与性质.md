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

你是单细胞测序组的工程师。一批数据是 **3000 个细胞 × 2000 个高变基因**的表达矩阵，fp32 占 $3000\times2000\times4\ \text{B}=24\ \text{MB}$。组里约定：低于 **$10^{-8}$** 的奇异值视为测序噪声，不算一个独立维度；这个阈值决定下游聚类保留几个主成分。

你按教科书流程先算 $A^\top A$（$2000\times2000$），再对它做特征分解、开方，得到最小奇异值 $3.2\times10^{-9}$，于是判定数值秩 1999。同事用 LAPACK 的 `svd` 直接算同一个矩阵，得到 $1.1\times10^{-6}$，数值秩 2000。两个数差了约 **350 倍**，而阈值 $10^{-8}$ 恰好卡在中间：一个说「这一维是噪声，截掉」，一个说「它是真实维度，保留」。

决策的代价不对称：截错了，可能把分开两个稀有亚群的方向当噪声删掉，聚类少一类，后续验证实验的钱白花；不截，噪声方向会污染降维结果。按一次单细胞建库约 **3 万元**估算，重做一轮既费钱又费样本。

问题出在第一步：$A^\top A$ 把条件数（condition number）**平方**了。读完你能回答三个问题：

1. 奇异值和 $A^\top A$ 特征值的确切关系是什么？
2. 为什么显式乘 $A^\top A$ 会丢掉一半有效数字？
3. 手算、证明和真正跑数值计算，应该分别用哪条路？

## 1. 直觉

$A^\top A$ 是「从输入侧看 $A$」：它记录每个输入方向被放大的**平方**。因为 $\|A\mathbf{v}\|^2=\mathbf{v}^\top A^\top A\mathbf{v}$，若 $\mathbf{v}$ 是 $A^\top A$ 的特征向量、特征值为 $\lambda$，则 $\|A\mathbf{v}\|^2=\lambda$，拉伸倍数是 $\sqrt\lambda$——这就是奇异值。所以 $A^\top A$ 的特征向量 = 右奇异向量，特征值 = $\sigma^2$。

对称地，$AA^\top$ 是「从输出侧看 $A$」，它的特征向量是左奇异向量，非零特征值同样是 $\sigma^2$。两张「X 光片」看的是同一副骨架。

> 💡 **一句话直觉**：**SVD = 对 $A^\top A$ 做谱分解，再开方；$U$ 由 $A$ 把 $V$ 搬过去得到。**

### 1.1 它在现实世界里对应什么

在单细胞数据里，$A$ 的每一行是一个细胞、每一列是一个基因。SVD 给出的第 $i$ 个三元组可以读成：

| 数学对象 | 单细胞数据里的对应物 |
|---|---|
| $\sigma_i$ | 第 $i$ 个「协同表达程序」的强度 |
| $\mathbf{v}_i$ | 这个程序在 2000 个基因上的权重 |
| $\mathbf{u}_i$ | 这个程序在 3000 个细胞上的得分 |
| $A^\top A$ | 从「基因-基因」侧拍的 X 光片 |
| $AA^\top$ | 从「细胞-细胞」侧拍的 X 光片 |

两张片子看的是同一副骨架：非零特征值都是 $\sigma_i^2$，只是一个按基因排、一个按细胞排。这也是为什么 $A^\top A$ 那么诱人——它把 $3000\times2000$ 的问题压成 $2000\times2000$。

### 1.2 这个类比在哪里失效

- SVD 给的是**正交的线性程序**；真实的基因调控有非线性、有层级，$\sigma_i$ 不能直接当因果强度读。
- 「X 光片」把对比度平方：$\lambda=\sigma^2$。$\sigma$ 从 $10^{0}$ 到 $10^{-8}$ 时，$\lambda$ 从 $10^{0}$ 到 $10^{-16}$，正好撞上双精度的颗粒（$2^{-52}\approx2.2\times10^{-16}$）。图像还在，细节已经糊了。
- 因此 $A^\top A$ 适合**手算和证明**，不适合**数值计算**；这两件事在这一节要分开说。

### 1.3 量级直觉

双精度的相对分辨率是 $\varepsilon_{\text{mach}}\approx2.2\times10^{-16}$。因为 $\lambda_i/\lambda_1=(\sigma_i/\sigma_1)^2$，要让它不被颗粒淹没，需要

$$
(\sigma_i/\sigma_1)^2>\varepsilon_{\text{mach}}
\;\Longrightarrow\;
\sigma_i/\sigma_1>\sqrt{\varepsilon_{\text{mach}}}\approx1.5\times10^{-8}.
$$

这个式子说的是：**显式构造 $A^\top A$ 的路线，只能可靠分辨到约 $10^{-8}$ 的相对奇异值**；再小就和 0 分不开。直接对 $A$ 做 SVD 没有这个平方，门槛回到 $10^{-16}$ 量级。

代价还有算力：$m=3000,n=2000$ 时，形成 $A^\top A$ 本身就要 $2mn^2=2.4\times10^{10}$ FLOP，和一次直接 SVD 的 $O(mn^2)$ 同量级；再对它做 $2000\times2000$ 的特征分解是额外开销。**精度更差，算力却没省。**

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

LAPACK 的 `gesdd`/`gesvd` 走的是**双对角化 + 隐式 QR**（[Golub–Kahan](#ref-golub-kahan-1965) 1965 年给出的算法）：先用 Householder 变换把 $A$ 化成上双对角矩阵（$O(mn^2)$），再迭代求奇异值。全程只对 $A$ 做正交变换，从不形成 $A^\top A$。NumPy 的 `np.linalg.svd`、PyTorch 的 `torch.linalg.svd` 都调用它。

## 3. 为什么这样定义

**为什么 $A^\top A$ 是半正定的？** 对任意 $\mathbf{x}$ 有 $\mathbf{x}^\top A^\top A\mathbf{x}=\|A\mathbf{x}\|_2^2\ge0$，二次型永不为负，所以特征值 $\lambda_i\ge0$，开方总合法。这是奇异值非负的代数根源。

**为什么用 $A^\top A$ 而不是随便一个对称矩阵？** 因为它恰好编码了「所有输入方向被拉伸的平方」，而谱定理保证它有完整的正交特征基。$A$ 本身可能没有实特征值（如旋转矩阵），但 $A^\top A$ 一定有，而且是对称的——这就是 SVD 比特征分解普适的原因。

**为什么还要补 $U$ 的剩余列？** 当 $m>r$ 时，$A\mathbf{v}_i/\sigma_i$ 只给出 $r$ 个方向，$U$ 需要 $m$ 个。缺的那些对应「任何输入都到不了的方向」，即左零空间，可以任意选一组标准正交基，它们不影响 $A=U\Sigma V^\top$（因为乘的 $\sigma$ 是 0）。

### 3.1 反事实：真的去乘 $A^\top A$ 会坏在哪

取一个能手算的 $2\times2$ 例子，$\varepsilon=10^{-8}$：

$$
A=\begin{bmatrix}1&1\\1&1+\varepsilon\end{bmatrix},\qquad
A^\top A=\begin{bmatrix}2&2+\varepsilon\\2+\varepsilon&2+2\varepsilon+\varepsilon^2\end{bmatrix}.
$$

它的迹约等于 $4$，行列式恰好是 $\varepsilon^2=10^{-16}$。于是

$$
\lambda_1\approx4,\qquad
\lambda_2=\frac{\det(A^\top A)}{\lambda_1}\approx\frac{10^{-16}}{4}=2.5\times10^{-17},
\qquad \sigma_2=\sqrt{\lambda_2}\approx5\times10^{-9}.
$$

到这里数学没错。坏在浮点：$A^\top A$ 的元素量级是 $2$，一次舍入的绝对误差约 $2\varepsilon_{\text{mach}}\approx4.4\times10^{-16}$；而你要分辨的 $\lambda_2$ 只有 $2.5\times10^{-17}$——**比单个元素的一次舍入还小一个数量级**。换句话说，

$$
\frac{\lambda_2}{\lambda_1}\approx6.25\times10^{-18}\;<\;\varepsilon_{\text{mach}}\approx2.2\times10^{-16},
$$

$\lambda_2$ 落在 $\lambda_1$ 的一个 ulp 以下，在算出来的矩阵里根本表示不出来：它可能被算成 $0$，也可能因舍入变成负数，开方直接得到 `nan`。

后果是数值秩判错：真值 $\sigma_2=5\times10^{-9}$，$A^\top A$ 路线可能报 $\sigma_2=0$，于是你把一个真实维度截掉。更讽刺的是，$\kappa_2(A)\approx4\times10^{8}$，而 $\kappa_2(A^\top A)\approx1.6\times10^{17}$，已经超过 $1/\varepsilon_{\text{mach}}\approx4.5\times10^{15}$——**连条件数本身都算不准了**。

对照直接 SVD：$\lvert\det A\rvert=\varepsilon$，$\sigma_1\sigma_2=\varepsilon$，$\sigma_1\approx2$，所以 $\sigma_2\approx5\times10^{-9}$，和上面一致，但全程只对 $A$ 做正交变换，不平方。**手算和证明用 $A^\top A$，数值计算用 LAPACK 的 SVD**——这不是风格问题，是有效数字问题。

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

### 5.3 场景闭环：3000×2000 的单细胞矩阵

**输入**：3000×2000 fp32，24 MB。每列已按测序深度归一化并中心化，否则第一主成分会指向整体表达水平。

**计算**：直接 SVD 的主项是双对角化，约 $O(mn^2)=1.2\times10^{10}$ FLOP 量级，单核按 10 GFLOP/s 估约 1.2 s。若先乘 $A^\top A$：形成 Gram 矩阵要 $2mn^2=2.4\times10^{10}$ FLOP，再对 2000×2000 做特征分解约 $O(n^3)=8\times10^{9}$ FLOP。两者的时间同量级，但后者多占 16 MB，还把 $\kappa$ 平方。

**输出**：数值秩 $r_\tau$（取 $\tau=10^{-8}$）与保留的主成分数 $k$（例如 30）。两条路线在 $\sigma_{\min}$ 上给出 $3.2\times10^{-9}$ 与 $1.1\times10^{-6}$，正好跨过阈值，于是 $r_\tau=1999$ 或 $2000$。

**成本**：完整 SVD 的 $U$ 是 $3000\times2000$ fp32=24 MB，$V$ 是 16 MB；只要前 30 个主成分时，$U_{30}$ 为 360 kB、$V_{30}$ 为 240 kB。$A^\top A$ 路线额外占 $2000^2\times4$ B=16 MB，却省不下时间。

**失败条件**：

- **阈值附近的奇异值**：$\sigma_i$ 落在 $10^{-8}$ 附近时，两种算法给出不同的秩，下游聚类可能少一个稀有亚群。兜底：不要只报秩，把谱画出来，并对 $\tau$ 做 $\pm10$ 倍敏感性分析；结论随 $\tau$ 翻转就说明这一维不可靠。
- **噪声不是白噪声**：测序深度、批次效应会集中在少数方向，截断可能删掉批次而不是噪声。兜底：先做批次校正，或把已知协变量回归掉。
- **精度天花板**：bf16 下相对精度约 $4\times10^{-3}$，$\sigma_i/\sigma_1<10^{-2}$ 的方向不可信；低精度流程里不要用 $A^\top A$ 判秩。

## 6. 常见误区

1. **误区**：奇异值是「$A$ 的特征值开平方」。
   **纠正**：是 $A^\top A$（或 $AA^\top$）特征值的平方根。$A$ 可能没有实特征值，$\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ 的特征值是 $\pm i$，奇异值却是 $1,1$。
2. **误区**：$\sigma_i=\lvert\lambda_i\rvert$ 总是成立。
   **纠正**：只对正规矩阵成立。$\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 特征值双 1，奇异值是 $1.618$ 与 $0.618$（黄金比例）。
3. **误区**：算出一个很小的奇异值就说明矩阵秩亏。
   **纠正**：先看阈值。NumPy 的 `matrix_rank` 用 $\sigma_i>\max(m,n)\cdot\varepsilon\cdot\sigma_1$；低于阈值的才算 0（数值秩，第 7.4 节）。这个数值秩判据在 [Golub 与 Van Loan](#ref-golub-van-loan-2013) 里有系统讨论。
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

**算错时下游看到什么**：若用 $A^\top A$ 路线把 $\sigma_{\min}$ 算成 $3.2\times10^{-9}$、据此判秩 1999，下游就少一个维度。如果那个方向恰好分开两个稀有亚群，聚类结果会少一类，标记基因列表里也找不到它。用户看到的是「细胞类型数比预期少一个」，而不是一个报错。兜底是同时打印谱、阈值和两种算法的差，并对 $\tau$ 附近的方向做敏感性分析。

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
- **物理意义 / 现实对应**：$\sigma_i$ 是第 $i$ 个方向的「透光率」；$A^\top A$ 的特征值是透光率的平方，所以它把动态范围平方，可分辨的 $\sigma$ 相对下限从 $\varepsilon_{\text{mach}}$ 抬到 $\sqrt{\varepsilon_{\text{mach}}}\approx1.5\times10^{-8}$。
- $A^\top A$ 与 $AA^\top$ 非零特征值相同，都是 $\sigma_i^2$。
- $\|A\|_2=\sigma_1$，$\|A\|_F^2=\sum\sigma_i^2$，$\kappa_2=\sigma_1/\sigma_r$，$\lvert\det A\rvert=\prod\sigma_i$。
- 特征分解只对方阵、可能有复数；SVD 对任意矩阵、一定非负；手算用 $A^\top A$，数值计算用 LAPACK，显式乘 $A^\top A$ 会把 $\kappa$ 平方。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-golub-kahan-1965"></a>**Golub 与 Kahan（Gene H. Golub & William Kahan, 1965）**：*Calculating the Singular Values and Pseudo-Inverse of a Matrix*。把 SVD 计算化成「Householder 双对角化 + QR 迭代」，全程不形成 $A^\top A$，因此不会把条件数平方；这就是本页第 2.3 节里 LAPACK `gesvd`/`gesdd` 的算法祖先。[SIAM J. Numer. Anal. 2(2)](https://doi.org/10.1137/0702016)
- <a id="ref-golub-van-loan-2013"></a>**Golub 与 Van Loan（Gene H. Golub & Charles F. Van Loan, 2013）**：*Matrix Computations*（4th ed.）。对 SVD 的存在性、数值秩、条件数与最小二乘给出标准处理，本页第 4.2 节的性质清单与第 6 节的秩判据都可在此查证。[Johns Hopkins University Press](https://doi.org/10.56021/9781421407944)

---

下一篇 → [7.3 低秩近似与 Eckart–Young](03-低秩近似.md)
