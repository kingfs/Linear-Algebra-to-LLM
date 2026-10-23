---
id: 7.1
title: SVD 的几何：旋转-拉伸-旋转
module: M07
slug: svd-geometry
prereq: [6.4]
minutes: 35
level: 核心
tags: [SVD, 奇异值, 正交矩阵, 几何]
llm_hooks: [weight-spectrum, spectral-norm]
---

# 7.1 SVD 的几何：旋转-拉伸-旋转

> **本节位置**：模块 7 · SVD ｜ **前置**：[6.4 对称矩阵与谱定理](../06-特征值/04-对称矩阵与谱定理.md) ｜ **预计**：35 分钟
>
> **一句话**：任何矩阵的作用都能拆成三步——先转、再沿坐标轴拉伸、再转；拉伸的倍数就是奇异值。

## 0. 场景引入

县档案馆要把 **12 万张** 1024×1024 的 8 位灰度老照片做成在线检索库。原始数据是 $12\times10^4\times1024^2\times1\ \text{B}\approx126\ \text{GB}$，可对象存储预算只批了 **25 GB**，必须压到原来的五分之一左右。

最省事的办法是降分辨率，但馆里明确要求「放大后还能看清印章边缘」。另一条路是把每张图当成矩阵 $A\in\mathbb{R}^{1024\times1024}$，只存前 $k$ 个奇异值和左右奇异向量：$k=50$ 时存 $50\times(1024+1024+1)=102{,}450$ 个数，是原来 $1{,}048{,}576$ 个像素的 $9.8\%$，看上去正好。

真正的决定不是「压不压」，而是**每张图取多大的 $k$**：取小了，印章边缘糊；取大了，预算爆。而且「丢掉的那部分到底损失了什么」必须能提前算出来，不能压完再肉眼比对。代价也很实在：原图不留底，压过头就不可逆。

先转、再沿两根正交轴拉伸、再转回去——这个几何图像不只属于压缩，它也是后面低秩近似、伪逆与 PCA 共用的骨架。

读完你能回答三个问题：

1. 为什么「先转、再拉伸、再转」三步能概括任意矩阵？
2. 奇异值凭什么可以当作「每层图案的音量」？按它排序漏掉了什么？
3. 一张 1024×1024 的图，前 50 个奇异值能保留多少能量，怎么估？

## 1. 直觉

先看最干净的情形 $A=\begin{bmatrix}3&0\\0&2\end{bmatrix}$。它把单位圆上的 $(x,y)$ 变成 $(3x,2y)$。页首的图正是这个画面：圆被拉成椭圆，半轴长 3 和 2 就是两个奇异值；这里没有旋转，因为 $A$ 已经沿坐标轴。

若 $A=\begin{bmatrix}3&0\\4&5\end{bmatrix}$，椭圆就**歪**了。但关键事实不变：**结果仍是椭圆**，只是两根轴不沿 $x$、$y$。

这个「单位球的像是超椭圆」的表述，[Trefethen 与 Bau](#ref-trefethen-1997) 第 4 讲给出了完整证明。

于是有三问：椭圆两根轴指向哪（两个**正交**的输出方向 $\mathbf{u}_1,\mathbf{u}_2$）？它们多长（非负数 $\sigma_1,\sigma_2$）？圆上哪两个**正交**方向被拉成了它们（$\mathbf{v}_1,\mathbf{v}_2$）？合起来就是

$$
A\underbrace{\begin{bmatrix}\uparrow&\uparrow\\ \mathbf{v}_1&\mathbf{v}_2\\ \downarrow&\downarrow\end{bmatrix}}_{V}
=\underbrace{\begin{bmatrix}\uparrow&\uparrow\\ \mathbf{u}_1&\mathbf{u}_2\\ \downarrow&\downarrow\end{bmatrix}}_{U}
\underbrace{\begin{bmatrix}\sigma_1&0\\0&\sigma_2\end{bmatrix}}_{\Sigma}
$$

> 💡 **一句话直觉**：**SVD 说「任何线性变换 = 找到一组输入正交方向，它们被拉伸成一组输出正交方向」。**

### 1.1 它在现实世界里对应什么

把 1024×1024 的图想成一摞**半透明胶片**：

| 数学对象 | 图像里的对应物 |
|---|---|
| $U$ 的第 $i$ 列 $\mathbf{u}_i$ | 第 $i$ 张胶片在「竖直方向」的条纹图案 |
| $V$ 的第 $i$ 列 $\mathbf{v}_i$ | 第 $i$ 张胶片在「水平方向」的条纹图案 |
| $\sigma_i$ | 这张胶片的曝光强度（音量） |
| $\mathbf{u}_i\mathbf{v}_i^\top$ | 一张秩 1 的「条纹图」 |
| 截断到 $k$ | 只洗前 $k$ 张胶片来叠 |

这个对应的内核是**叠加**：把每张条纹图按 $\sigma_i$ 曝光、再逐像素相加，就得到原图。图像压缩、推荐系统的潜在因子、CT 重建，用的是同一摞胶片，只是「行」和「列」换了含义。

### 1.2 这个类比在哪里失效

- 胶片叠加是线性、可交换的；人眼不是。第 10 张胶片可能只占 0.1% 能量，却正好是印章的边缘。
- JPEG 用的是**固定**的 DCT 基，SVD 的基是**每张图自己算的**：你必须把 $U_k,V_k$ 一并存下来，否则解不出图。
- 「旋转」包含**反射**（把纸翻面），所以准确说法是「正交变换—拉伸—正交变换」；二维反射的行列式为 $-1$，肉眼看不出转了多少，但它仍是合法的正交变换。

### 1.3 量级直觉

一张 1024×1024 的图最多有 1024 个奇异值。$k=50$ 只留下 **4.9%** 的奇异值个数，却可能留下绝大部分能量。具体多少取决于谱：若按 $\sigma_i\propto1/i$ 衰减，则

$$
\eta_{50}=\frac{\sum_{i=1}^{50}i^{-2}}{\sum_{i=1}^{1024}i^{-2}}
=\frac{1.6251}{1.6440}\approx98.9\%.
$$

这个式子说的是：**按 $1/i$ 衰减时，前 50 个方向就占了约 99% 的 Frobenius 能量**。真实照片的谱衰减由内容决定，可能更快（大面积平滑）也可能更慢（纹理、噪声），必须对每张图实测，不能套这个数。

## 2. 定义

### 2.1 正交矩阵

方阵 $Q\in\mathbb{R}^{n\times n}$ 若满足 $Q^\top Q=QQ^\top=I_n$，就称为**正交矩阵（orthogonal matrix）**，等价于「列向量两两正交且长度为 1」。几何上它保持内积与长度：$\|Q\mathbf{x}\|_2=\|\mathbf{x}\|_2$，把球面映成球面，只改朝向、不拉伸。

### 2.2 奇异值分解

设 $A\in\mathbb{R}^{m\times n}$，$r=\operatorname{rank}(A)$。则存在正交矩阵 $U\in\mathbb{R}^{m\times m}$、$V\in\mathbb{R}^{n\times n}$ 及对角矩阵 $\Sigma\in\mathbb{R}^{m\times n}$，使 $A=U\Sigma V^\top$。

$\Sigma$ 只有 $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0$ 非零，其后全为 0。$\sigma_i$ 称**奇异值（singular value）**，$U$ 的列称**左奇异向量（left singular vectors）**，$V$ 的列称**右奇异向量（right singular vectors）**。形状先记牢，这是最容易错的地方：

| 对象 | 形状 | 含义 |
|---|---|---|
| $A$ | $m\times n$ | 任意实矩阵 |
| $U$ | $m\times m$ | 输出空间的正交基 |
| $\Sigma$ | $m\times n$ | 与 $A$ 同形，对角线上是 $\sigma_i$ |
| $V$ | $n\times n$ | 输入空间的正交基 |

### 2.3 瘦 SVD 与三步读法

实际计算和存储常去掉零部分，只留前 $r$ 项：

$$
A=\sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top
$$

这叫**瘦 SVD（thin SVD）**，每项是一个秩 1 矩阵，权重为 $\sigma_i$；NumPy 的 `full_matrices=False` 给的就是它。把 $A\mathbf{x}$ 按 $V^\top\to\Sigma\to U$ 拆开：先把输入换到奇异向量坐标系，再沿每根轴乘 $\sigma_i$，最后转回原坐标系。**只有中间一步改变长度**。

## 3. 为什么这样定义

**为什么必须正交？** 若允许任意可逆矩阵，任何矩阵都能写成 $P\Lambda Q^{-1}$，这个式子毫无信息量。正交约束要求：输入的探测方向互不干扰、输出的结果方向互不干扰、中间耦合全部消失，只剩「各自乘一个数」。此外正交变换不放大误差，于是「拉伸」被干净地隔离在 $\Sigma$ 里，分析稳定性时只看 $\sigma_i$。

**为什么不叫特征分解？** 特征分解（eigendecomposition）$A=P\Lambda P^{-1}$ 要求 $A$ 是方阵且可对角化，$\Lambda$ 可以是复数或负数。SVD 对**任意** $m\times n$ 矩阵存在，$\sigma_i$ 一定是非负实数。代价是 SVD 用**两组**基（$U$ 和 $V$），特征分解只用一组：SVD 描述的是「输入、输出各自换基后的对角形」，不是同一个空间里的换基。

**它从哪里来？** 预告（第 7.2 节展开）：$A^\top A$ 是 $n\times n$ 对称半正定矩阵，谱定理保证它有正交特征基，特征值 $\lambda_i\ge0$；取 $\sigma_i=\sqrt{\lambda_i}$、$V$ 的列为特征向量，再由 $\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$ 定义左奇异向量，就得到 SVD。

更早的历史线索是：Beltrami（1873）与 Jordan（1874）分别对实、复方阵给出了这种分解，之后才推广到长方形矩阵；[Stewart 的综述](#ref-stewart-1993) 把这段谱系考据得很清楚。

### 3.1 反事实：按特征值大小排序会坏在哪

一个很自然的手滑，是把 SVD 当成「特征值分解的推广」，于是按 $|\lambda_i|$ 从大到小排序、保留前 $k$ 个。取

$$
A=\begin{bmatrix}0&1\\0&0\end{bmatrix}.
$$

它的特征多项式是 $\lambda^2=0$，两个特征值都是 $0$。按特征值排序，你会认为「它什么都不做」，截断成 $B=0$，误差 $\|A\|_F=1$。

但 $A$ 明明把 $\mathbf{e}_2$ 搬成 $\mathbf{e}_1$：$\|A(0,1)^\top\|=1$。它的奇异值是 $\sigma_1=1,\sigma_2=0$，所以 $A_1=A$，误差 $0$。**特征值看的是「同一个方向进、同一个方向出」，SVD 看的是「一个方向进、另一个方向出」；非正规矩阵里这两件事可以完全脱节。**

再看一个不那么极端的例子。$A=\begin{bmatrix}3&0\\4&5\end{bmatrix}$ 的特征值是 $3$ 和 $5$（上三角），而奇异值是 $6.708$ 和 $2.236$——**排序都不一样**。按特征值保留 $\lambda=5$ 对应的方向 $\mathbf{e}_2$，得到 $B=\begin{bmatrix}0&0\\0&5\end{bmatrix}$，误差

$$
\|A-B\|_F=\left\|\begin{bmatrix}3&0\\4&0\end{bmatrix}\right\|_F=5.
$$

按奇异值排序的 $A_1$ 误差只有 $\sigma_2=\sqrt5\approx2.236$，小 $2.2$ 倍。坏的不是「算了半天」，而是**选错了方向**：$A$ 真正拉得最狠的输入方向是 $(1,1)/\sqrt2$，不是 $\mathbf{e}_2$。

## 4. 性质与推导

### 4.1 核心性质

| 性质 | 式子 | 读法 |
|---|---|---|
| 最大拉伸 | $\|A\mathbf{x}\|_2\le\sigma_1\|\mathbf{x}\|_2$ | 单位向量最多被拉 $\sigma_1$ 倍，取等当 $\mathbf{x}\parallel\mathbf{v}_1$ |
| 最小拉伸 | $\|A\mathbf{x}\|_2\ge\sigma_n\|\mathbf{x}\|_2$ | 仅当 $A$ 可逆；最坏方向被压到 $\sigma_n$ |
| 体积 | $\lvert\det A\rvert=\prod_i\sigma_i$ | 各方向拉伸倍数相乘（方阵） |
| 秩 | $\operatorname{rank}(A)=\#\{i:\sigma_i>0\}$ | 零奇异值个数 = 被压没的维数 |
| 转置 | $A^\top=V\Sigma^\top U^\top$ | $A$ 与 $A^\top$ 共享奇异值，$U,V$ 互换 |
| 正交不变 | $\sigma_i(QA)=\sigma_i(A)$ | 左边乘正交矩阵不改变拉伸 |

### 4.2 推导：$\sigma_1$ 是最大放大倍数

在约束 $\|\mathbf{x}\|=1$ 下最大化 $\|A\mathbf{x}\|^2=\mathbf{x}^\top A^\top A\mathbf{x}$，拉格朗日条件是 $A^\top A\mathbf{x}=\lambda\mathbf{x}$。这句话的意思是：**取极值的方向必须是 $A^\top A$ 的特征向量**。此时 $\|A\mathbf{x}\|^2=\lambda$，故最大放大倍数为 $\sqrt{\lambda_{\max}(A^\top A)}=\sigma_1$，最小值为 $\sqrt{\lambda_{\min}}$。$A^\top A$ 半正定给出 $\lambda_i\ge0$，开方总是合法——这就是奇异值非负的来源。

### 4.3 推导：为什么 $\mathbf{u}_i$ 自动正交

设 $\mathbf{v}_i,\mathbf{v}_j$ 是 $A^\top A$ 属于不同特征值的单位特征向量，则 $\mathbf{v}_i^\top\mathbf{v}_j=0$，于是

$$
\mathbf{u}_i^\top\mathbf{u}_j=\frac{(A\mathbf{v}_i)^\top(A\mathbf{v}_j)}{\sigma_i\sigma_j}
=\frac{\mathbf{v}_i^\top A^\top A\mathbf{v}_j}{\sigma_i\sigma_j}
=\frac{\lambda_j\,\mathbf{v}_i^\top\mathbf{v}_j}{\sigma_i\sigma_j}=0
$$

即 $A$ 把正交的输入方向映成**仍然正交**的输出方向。拉伸不产生「串扰」，正是 SVD 能成立的关键；把单位球代入三步也立刻看出：球 → 球 → 椭球（半轴 $\sigma_i$）→ 转向的椭球。

## 5. 例子

### 5.1 手算：$2\times2$ 的完整 SVD

$$
A=\begin{bmatrix}3&0\\4&5\end{bmatrix},\qquad
A^\top A=\begin{bmatrix}3&4\\0&5\end{bmatrix}\begin{bmatrix}3&0\\4&5\end{bmatrix}
=\begin{bmatrix}25&20\\20&25\end{bmatrix}
$$

**特征值**：$\det(A^\top A-\lambda I)=(25-\lambda)^2-400=0$，得 $\lambda_1=45$、$\lambda_2=5$。

**右奇异向量**：$\lambda_1=45$ 时 $-20x+20y=0\Rightarrow x=y$，取 $\mathbf{v}_1=\frac{1}{\sqrt2}(1,1)^\top$；$\lambda_2=5$ 时 $20x+20y=0\Rightarrow x=-y$，取 $\mathbf{v}_2=\frac{1}{\sqrt2}(1,-1)^\top$。两者正交，符合谱定理。

**奇异值**：$\sigma_1=\sqrt{45}=3\sqrt5\approx6.708$，$\sigma_2=\sqrt5\approx2.236$。

**左奇异向量**：由 $\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$ 得

$$
\mathbf{u}_1=\frac{(3,9)/\sqrt2}{3\sqrt5}=\frac{1}{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix},
\qquad
\mathbf{u}_2=\frac{(3,-1)/\sqrt2}{\sqrt5}=\frac{1}{\sqrt{10}}\begin{bmatrix}3\\-1\end{bmatrix}
$$

**核对**：$\mathbf{u}_1^\top\mathbf{u}_2=(3-3)/10=0$ ✓；$\det A=15=\sigma_1\sigma_2=3\sqrt5\cdot\sqrt5$ ✓；重建时 $\sigma_1\mathbf{u}_1\mathbf{v}_1^\top=\frac32\begin{bmatrix}1&1\\3&3\end{bmatrix}$，$\sigma_2\mathbf{u}_2\mathbf{v}_2^\top=\frac12\begin{bmatrix}3&-3\\-1&1\end{bmatrix}$，相加正是 $\begin{bmatrix}3&0\\4&5\end{bmatrix}$ ✓

**几何图像**：单位圆 → 椭圆，半轴 $6.708$（沿 $\mathbf{u}_1\approx(0.316,0.949)$，约 $71.6^\circ$）与 $2.236$；输入侧被拉得最狠的方向是 $\mathbf{v}_1$（$45^\circ$ 对角线）。面积由 $\pi$ 变 $15\pi\approx47.1$。NumPy 会给出 $\mathbf{u}_1,\mathbf{v}_1$ 整体取负，这是符号自由度，重建结果相同，别当成算错。

### 5.2 场景：非方阵把球压成椭圆盘

$A=\begin{bmatrix}1&1&0\\0&1&1\end{bmatrix}$ 把 $\mathbb{R}^3$ 映到 $\mathbb{R}^2$。$AA^\top=\begin{bmatrix}2&1\\1&2\end{bmatrix}$，特征值 $3,1$，故 $\sigma_1=\sqrt3$、$\sigma_2=1$。右奇异向量 $\mathbf{v}_1=\frac{1}{\sqrt6}(1,2,1)^\top$、$\mathbf{v}_2=\frac{1}{\sqrt2}(1,0,-1)^\top$，第三个 $\mathbf{v}_3=\frac{1}{\sqrt3}(-1,1,-1)^\top$ 对应 $\sigma_3=0$。

所以三维单位球 → 平面上的椭圆（半轴 $\sqrt3$ 和 1），沿 $\mathbf{v}_3$ 的分量被**完全压没**。这就是零空间的几何含义：3 维输入只有 2 个自由度活下来。

### 5.3 场景闭环：12 万张图、25 GB 预算

**输入**：12 万张 1024×1024 的 8 位灰度图，共 $1.258\times10^{11}$ B ≈126 GB。SVD 逐张做，单张按 fp32 占 $1024^2\times4$ B=4 MB，峰值内存可控。

**计算**：完整 SVD 约 $O(n^3)\approx1.07\times10^9$ FLOP/张，12 万张合计 $1.29\times10^{14}$ FLOP。单核按 10 GFLOP/s 估约 $1.29\times10^4$ s，即 **3.6 小时**；32 核按线性加速估约 7 分钟。

**输出**：每张图一个 $k$（按 $\eta_k\ge0.99$ 或人工抽检定），存 $U_k,\Sigma_k,V_k$。决策是「这张 $k=48$，那张 $k=64$」，不是给全库一个统一 $k$。

**成本**：$k=50$ 时数字个数是 102,450，只有原来的 9.8%；但字节账要重算：

| 存储格式 | 每张字节 | 相对原图 | 12 万张总量 |
|---|---|---|---|
| fp32 因子 | 409,800 B | 39% | 约 49 GB |
| fp16 因子 | 204,900 B | 19.5% | 约 24.6 GB |
| int8 因子 | 102,450 B | 9.8% | 约 12.3 GB |

（int8 一行按 1 B/数估，不含每列一个缩放因子。）

「9.8% 压缩」说的是**数的个数**，不是**字节**。只按个数报预算，会低估 2–4 倍，取决于因子用 fp16 还是 fp32。

**失败条件**：

- **谱衰减慢的图**：纹理、噪声多的照片，前 50 个奇异值可能只占 70% 能量，印章边缘糊。兜底：逐图跑 $\eta_k$ 曲线，不达标的图提高 $k$ 或保留原图。
- **只报平均压缩比**：预算按平均 $k$ 算，尾部几张高 $k$ 的图会把总量顶爆。兜底：按最坏图（最大 $k$）估算总量。
- **符号与基不唯一**：NumPy 给出的 $U,V$ 可能整体变号，重建不变；但把 $U_k,V_k$ 分开量化、再分别反量化，误差不再抵消。兜底：存完立刻重建一遍，按像素比对。

## 6. 常见误区

1. **误区**：$U$ 的列就是 $A$ 的特征向量。
   **纠正**：只有 $A$ 对称半正定时才可能。$A=\begin{bmatrix}0&1\\0&0\end{bmatrix}$ 特征值全为 0，奇异值却是 $1,0$，且 $\mathbf{u}_1=(1,0)$、$\mathbf{v}_1=(0,1)$。
2. **误区**：奇异值 = 特征值的绝对值。
   **纠正**：只对**正规矩阵**（$A^\top A=AA^\top$）成立。$A=\begin{bmatrix}1&1\\0&0\end{bmatrix}$ 特征值为 $1,0$，奇异值为 $\sqrt2,0$。
3. **误区**：旋转矩阵特征值是虚数，所以「没法分解」；或 SVD 唯一。
   **纠正**：$\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ 的奇异值是 $1,1$，它本身就是正交矩阵，SVD 是 $A=AI I$，从不要求特征值存在。另外 SVD 不唯一：$\mathbf{u}_i,\mathbf{v}_i$ 同时变号不改变 $A$，重奇异值对应子空间内可任选正交基。

## 7. 概念网络

- 前置：[6.4 对称矩阵与谱定理](../06-特征值/04-对称矩阵与谱定理.md) —— $A^\top A$ 对称，保证有正交特征基，这是 SVD 的存在性来源。
- 前置：[2.6 秩](../02-向量空间/06-秩.md) —— 秩 = 非零奇异值个数；[1.6 正交与投影](../01-向量与几何/06-正交与投影.md) —— 正交变换不改变长度。
- 后续：[7.2 计算与性质](02-SVD的计算与性质.md)、[7.3 低秩近似](03-低秩近似.md)、[7.4 伪逆与数值秩](04-伪逆与数值秩.md)。
- 关联：[5.1 行列式的几何](../05-行列式/01-行列式几何.md) —— $\lvert\det A\rvert=\prod\sigma_i$。

## 8. 通往大模型：权重矩阵的「骨架」

**出现位置**：每层线性投影 $W\in\mathbb{R}^{D\times D}$（$D=4096$）、注意力里的 $W_Q,W_K,W_V,W_O$、LoRA 的增量 $\Delta W=BA$。

**形状**：$W=U\Sigma V^\top$ 中 $U,\Sigma,V$ 均为 $4096\times4096$（瘦版只存 $r$ 列）。完整 SVD 是 $O(D^3)$，实际只求前几十个奇异值（截断 SVD，如 `scipy.sparse.linalg.svds`）。

**为什么需要**：谱范数（spectral norm）$\|W\|_2=\sigma_1$ 就是这一层的 Lipschitz 常数。对任意输入，

$$
\|W\mathbf{x}-W\mathbf{x}'\|_2\le\sigma_1\|\mathbf{x}-\mathbf{x}'\|_2
$$

这个式子说的是：$\sigma_1$ 是「输入扰动被放大的最坏倍数」。它连乘多层后大于 1，激活会指数增长；小于 1 则逐层衰减。$\sigma_{\min}$ 接近 0 时这一层几乎不可逆，微调梯度会沿那些方向爆炸；第 7.4 节的条件数（condition number）$\kappa=\sigma_1/\sigma_{\min}$ 会量化这个放大。

**LoRA 的形状**：$\Delta W=BA$，$B\in\mathbb{R}^{D\times r}$、$A\in\mathbb{R}^{r\times D}$，参数量 $2Dr$。$D=4096,r=16$ 时为 $131{,}072$，占全参数 $D^2=16{,}777{,}216$ 的 $0.78\%$。$\Delta W$ 至多有 $r$ 个非零奇异值，这就是「低秩更新」的几何解释（第 7.3 节）。

**可检验结论**：随机初始化的 $W$（元素方差 $1/D$）其奇异值落在 $[0,2]$ 区间、$\sigma_1\approx2$（Marchenko–Pastur 律），$\kappa$ 随 $D$ 增长而不是 1；若某层谱**快速衰减**（如 $\sigma_{50}/\sigma_1<0.01$），说明它高度冗余，可用秩 50 近似而几乎不损失能量。

```python
s = np.linalg.svd(W, compute_uv=False)   # W: [4096, 4096]
print(s[0], s[-1], s[0] / s[-1])
```

**算错时下游看到什么**：若把特征值当奇异值排序，选出的「主方向」可能完全错——$\begin{bmatrix}0&1\\0&0\end{bmatrix}$ 会被当成零矩阵，重建图出现本不该有的重影，或者把真实边缘当噪声抹掉。用户看到的是印章上的字糊成一团、OCR 拒识，而压缩率报表上一切正常。兜底是保留原图，并用 $\eta_k$ 与人工抽检两道门槛。

## 9. 练习

1. **概念**：为什么 SVD 要求 $U,V$ 正交，而不是任意可逆矩阵？从「信息量」角度说一句。
2. **计算**：求 $A=\begin{bmatrix}2&0\\0&-3\end{bmatrix}$ 与 $B=\begin{bmatrix}1&1\\1&1\end{bmatrix}$ 的 SVD（奇异值、左右奇异向量），并分别解释 $U$ 的负号与 $\sigma_2=0$ 的几何含义。
3. **场景**：某层奇异值为 $(10,1,0.01)$，输入扰动 $\|\delta\mathbf{x}\|=10^{-3}$，输出扰动最大多少？哪个方向最危险？
4. **开放**：只保留 $\sigma_1$ 会丢失哪类信息？对「分类」与「生成」两种任务，损失一样吗？

<details>
<summary>参考答案</summary>

1. 任意可逆矩阵有 $2n^2$ 个自由度，任何矩阵都能被「解释」成对角形；正交约束把自由度压到最少，使 $\Sigma$ 成为唯一携带拉伸信息的对象，且数值上不放大误差。
2. $A^\top A=\operatorname{diag}(4,9)$，按降序 $\sigma=(3,2)$：$\mathbf{v}_1=(0,1)$、$\mathbf{v}_2=(1,0)$；$\mathbf{u}_1=A\mathbf{v}_1/3=(0,-1)$、$\mathbf{u}_2=(1,0)$。故 $U=\begin{bmatrix}0&1\\-1&0\end{bmatrix}$、$\Sigma=\operatorname{diag}(3,2)$、$V=\begin{bmatrix}0&1\\1&0\end{bmatrix}$，负号来自 $A$ 把 $+y$ 翻到 $-y$（反射）。对 $B$：$B^\top B=\begin{bmatrix}2&2\\2&2\end{bmatrix}$，特征值 $4,0$，$\sigma=(2,0)$；$\mathbf{v}_1=(1,1)/\sqrt2$、$\mathbf{v}_2=(1,-1)/\sqrt2$、$\mathbf{u}_1=(1,1)/\sqrt2$、$\mathbf{u}_2=(1,-1)/\sqrt2$。$\sigma_2=0$ 表示沿 $\mathbf{v}_2$ 的输入全被压成 0，秩为 1。
3. 最大 $10\times10^{-3}=10^{-2}$，沿 $\mathbf{v}_1$；$\sigma_3=0.01$ 的方向几乎不传扰动，但那里的信息也不可恢复。
4. 丢掉除最大拉伸方向外的全部方向。分类常只需少数判别方向，损失可能很小；生成需要细节与多样性，单方向近似通常不可接受。

</details>

## 10. 一页速记

- $A=U\Sigma V^\top$：正交—拉伸—正交，对**任意** $m\times n$ 矩阵成立。
- **物理意义 / 现实对应**：SVD 是一台「先转正、再沿轴拉伸、再转回去」的机器；在图像里就是「把画面拆成一摞按音量排序的条纹胶片」，$\sigma_i$ 是每张胶片的曝光强度。
- $\sigma_i\ge0$ 是椭球半轴长；$\operatorname{rank}=\#\{\sigma_i>0\}$；$\lvert\det A\rvert=\prod\sigma_i$（方阵）。
- $\|A\mathbf{x}\|\le\sigma_1\|\mathbf{x}\|$，等号在 $\mathbf{x}\parallel\mathbf{v}_1$；$\sigma_1$ 就是谱范数。
- $A$ 与 $A^\top$ 共享奇异值，$\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$ 自动正交；符号与重奇异值带来不唯一性，NumPy 差个负号是正常的。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-stewart-1993"></a>**斯图尔特（G. W. Stewart, 1993）**：*On the Early History of the Singular Value Decomposition*。考据了 SVD 的早期谱系——Beltrami（1873）与 Jordan（1874）先对实、复方阵给出分解，此后才推广到长方形矩阵，正是本页第 3 节「它从哪里来」的时间线依据。[SIAM Review 35(4)](https://doi.org/10.1137/1035134)
- <a id="ref-trefethen-1997"></a>**Trefethen 与 Bau（Lloyd N. Trefethen & David Bau III, 1997）**：*Numerical Linear Algebra*。第 4 讲把 SVD 讲成「单位球的像是超椭圆，半轴长就是奇异值」，并用 Theorem 4.1 证明任意 $m\times n$ 矩阵都有 SVD，是本页第 1–2 节几何图像与存在性的直接来源。[SIAM, 1997](https://doi.org/10.1137/1.9780898719574)

---

下一篇 → [7.2 SVD 的计算与性质](02-SVD的计算与性质.md)
