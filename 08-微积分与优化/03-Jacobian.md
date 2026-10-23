---
id: 8.3
title: Jacobian：多元映射的导数
module: M08
slug: jacobian
prereq: [3.3, 8.1]
minutes: 40
level: 核心
tags: [Jacobian, 局部线性化, 链式法则]
llm_hooks: [backpropagation, autograd]
---

# 8.3 Jacobian：多元映射的导数

> **本节位置**：模块 8 · 微积分与优化 ｜ **前置**：[3.3 矩阵乘法](../03-矩阵与线性映射/03-矩阵乘法.md)、[8.1 梯度](01-多元函数与梯度.md) ｜ **预计**：40 分钟
>
> **一句话**：向量值函数的导数不再是一个数，而是一个矩阵——它把「输入动一点」线性地翻译成「输出动一点」。

## 0. 场景引入

一层神经网络做 $ \mathbf{z}=W\mathbf{x}+\mathbf{b} $，输入 $\mathbf{x}\in\mathbb{R}^{768}$，输出 $\mathbf{z}\in\mathbb{R}^{3072}$。

如果输入的第 3 个分量增加 $0.01$，输出会怎么变？如果 768 个输入同时动，3072 个输出各自变多少？

这需要 $3072\times768$ 个数：每个输出对每个输入的偏导。把它们排成矩阵，就是 [Jacobian（雅可比矩阵）](#ref-jacobi-1841)。**它不是人为构造的怪物，而是「线性映射的矩阵表示」在微积分里的自然出现。**

## 1. 直觉

### 1.1 局部线性化（local linearization）

一元函数在一点附近可以用切线近似；多元标量函数用切平面；向量值函数则用**线性映射**近似。

输入空间里一个很小的球，经过 $F$ 后变成一个很小的椭球（或压扁的平行四边形）。Jacobian 就是描述这个「球变椭球」的线性变换矩阵。

### 1.2 每一行是一个梯度

$F$ 的第 $i$ 个输出 $F_i$ 是一个标量函数。它的梯度 $\nabla F_i$ 是行向量（或列向量，取决于约定）。把 $m$ 个输出的梯度摞起来，就是 $m\times n$ 的 Jacobian。

> 💡 **一句话直觉**：**Jacobian = 把每个输出分量的梯度按行叠起来；它是一阶导数的「矩阵版」。**

类比边界：局部线性化只在一点附近有效；离得远时高阶项不可忽略，就像切线不能代替整条曲线。

## 2. 定义

设 $F:\mathbb{R}^n\to\mathbb{R}^m$，写成

$$
F(\mathbf{x})=\begin{bmatrix}F_1(x_1,\dots,x_n)\\\vdots\\F_m(x_1,\dots,x_n)\end{bmatrix}
$$

若所有偏导存在，定义 Jacobian 为

$$
J_F(\mathbf{x})=
\begin{bmatrix}
\dfrac{\partial F_1}{\partial x_1}&\cdots&\dfrac{\partial F_1}{\partial x_n}\\
\vdots&\ddots&\vdots\\
\dfrac{\partial F_m}{\partial x_1}&\cdots&\dfrac{\partial F_m}{\partial x_n}
\end{bmatrix}\in\mathbb{R}^{m\times n}
$$

即 $(J_F)_{ij}=\partial F_i/\partial x_j$。**行对应输出，列对应输入**，形状是 $[m,n]$。

可微的定义相应地写成 $F(\mathbf{x}+\Delta\mathbf{x})=F(\mathbf{x})+J_F(\mathbf{x})\,\Delta\mathbf{x}+o(\|\Delta\mathbf{x}\|)$。逐项读法：输出的变化约等于矩阵乘上输入的变化。

## 3. 为什么这样定义

标量函数的导数是一个数，因为「输入方向」只有两个（正、负），变化率用一个数就够。向量值函数不同：

- 输入有 $n$ 个独立方向；
- 输出有 $m$ 个分量，每个分量对每个输入方向都有各自的敏感度。

所以导数必须是一个**从输入空间到输出空间的线性映射**。选定标准基后，线性映射就是 $m\times n$ 矩阵。这不是选择，而是「线性映射 = 矩阵」的必然结果（3.3 节）。

如果只算每个输出的梯度而不用矩阵组织，链式法则会变成一堆下标；写成矩阵后，复合函数的导数就是矩阵乘法。这正是反向传播能用 GEMM 高效实现的原因。

## 4. 性质与推导

### 4.1 线性映射的 Jacobian 就是它自己

若 $F(\mathbf{x})=A\mathbf{x}$，则 $F_i=\sum_j A_{ij}x_j$，于是 $\partial F_i/\partial x_j=A_{ij}$，即 $J_F=A$。对仿射映射 $F(\mathbf{x})=A\mathbf{x}+\mathbf{b}$，同样有 $J_F=A$（常数的导数为零）。**线性函数是自身的导数**，这是后面「$y=Ax$ 的 Jacobian 是 $A$」的来源。

### 4.2 链式法则

若 $F:\mathbb{R}^n\to\mathbb{R}^m$、$G:\mathbb{R}^m\to\mathbb{R}^p$，则 $G\circ F$ 的 Jacobian 是 $J_{G\circ F}(\mathbf{x})=J_G(F(\mathbf{x}))\;J_F(\mathbf{x})$。

形状检查：$[p,m]\times[m,n]=[p,n]$，正好是复合映射的形状。**中间维数 $m$ 被消掉**，这就是「前向传播的形状链」，也是反向传播里梯度能逐层回传的原因。

### 4.3 坐标变换与行列式

极坐标 $F(r,\theta)=(r\cos\theta,\;r\sin\theta)$：

$$
J_F=\begin{bmatrix}\cos\theta&-r\sin\theta\\ \sin\theta&r\cos\theta\end{bmatrix},
\qquad \det J_F=r
$$

$\det J$ 是面积缩放因子（5.1 节）。$r=0$ 时行列式为 0，映射退化——极点在极坐标下确实「压扁」了。这也是变量替换公式里要乘 $|\det J|$ 的原因。

### 4.4 复合的顺序不能反

$J_GJ_F$ 与 $J_FJ_G$ 通常形状都不同；即使都是方阵，矩阵乘法也不可交换。链式法则的顺序由「先作用谁」决定，写反了就是另一个函数。

## 5. 例子

### 5.1 手算：二维映射的 Jacobian

$$
F(x,y)=\begin{bmatrix}x^2y\\ x+3y\end{bmatrix}
$$

偏导：

$$
J_F(x,y)=\begin{bmatrix}2xy&x^2\\1&3\end{bmatrix}
$$

在 $(1,2)$ 处：

$$
J_F(1,2)=\begin{bmatrix}4&1\\1&3\end{bmatrix},\qquad \det J=12-1=11
$$

$F(1,2)=(2,7)^\top$。用 Jacobian 预测 $F(1.1,\,1.9)$：

$$
\Delta\mathbf{x}=(0.1,-0.1)^\top,\quad
J\Delta=\begin{bmatrix}4&1\\1&3\end{bmatrix}\begin{bmatrix}0.1\\-0.1\end{bmatrix}
=\begin{bmatrix}0.3\\-0.2\end{bmatrix}
$$

所以 $F\approx(2.3,\,6.8)^\top$。精确值：$1.21\times1.9=2.299$，$1.1+5.7=6.8$。预测几乎完全正确，因为步长很小，二阶项可以忽略。

### 5.2 手算：极坐标在 $(r,\theta)=(2,\pi/6)$

$$
J=\begin{bmatrix}\cos30^\circ&-2\sin30^\circ\\ \sin30^\circ&2\cos30^\circ\end{bmatrix}
=\begin{bmatrix}0.866&-1\\0.5&1.732\end{bmatrix}
$$

$\det J=0.866\times1.732-(-1)\times0.5=1.5+0.5=2=r$。面积放大了 2 倍：极坐标下 $r\,\mathrm{d}r\,\mathrm{d}\theta$ 的那个 $r$，就是这里来的。

### 5.3 场景：两层网络的逐样本 Jacobian

设 $\mathbf{z}=W_2\,\sigma(W_1\mathbf{x})$，其中 $W_1\in\mathbb{R}^{h\times d}$、$W_2\in\mathbb{R}^{o\times h}$。由链式法则：

$$
J_{\mathbf{z}}(\mathbf{x})=W_2\;\operatorname{diag}\big(\sigma'(\cdot)\big)\;W_1
$$

形状 $[o,h]\times[h,h]\times[h,d]=[o,d]$。它告诉你「这个样本的输入像素动一点，输出 logits 动多少」。实际训练不会把它算出来，但反向传播传的正是它的转置作用（见 §8）。

## 6. 常见误区

1. **误区**：Jacobian 一定是 $n\times m$。
   **纠正**：形状取决于约定。本文统一「行 = 输出、列 = 输入」，即 $[m,n]$。别的书写成 $[n,m]$ 是其转置，公式里的转置位置会不同。

2. **误区**：Jacobian 的每个元素就是「总导数」。
   **纠正**：$J_{ij}$ 是**偏导**，固定其它输入。若变量之间还有依赖（如 $x=x(t)$），总导数要把间接路径加起来，那才是链式法则。

3. **误区**：Jacobian 是常数矩阵。
   **纠正**：只有线性/仿射映射的 Jacobian 处处相同。一般情况下它随 $\mathbf{x}$ 变化，例如 §5.1 的 $J$ 依赖 $x,y$。

4. **误区**：$\det J=0$ 说明函数不可逆。
   **纠正**：只说明逆函数定理不能直接使用，该点附近可能压扁或折叠；不能推出函数处处不可逆。例如 $f(x)=x^3$ 在 $0$ 处导数为 $0$，但全局可逆。

## 7. 概念网络

- 前置：[3.3 矩阵乘法](../03-矩阵与线性映射/03-矩阵乘法.md) —— Jacobian 是线性映射的矩阵。
- 前置：[8.1 梯度](01-多元函数与梯度.md) —— 每一行是一个输出的梯度。
- 后续：[8.4 Hessian](04-Hessian与二阶近似.md) —— 梯度的 Jacobian 就是 Hessian。
- 后续：[8.5 矩阵求导与链式法则](05-矩阵求导与链式法则.md) —— 用 Jacobian 组织反向传播。
- 关联：5.1 行列式 —— $\det J$ 是局部体积缩放；变量替换、归一化流都靠它。

## 8. 通往大模型：反向传播从不真的造 Jacobian

**出现位置**：autograd 的反向模式（reverse-mode AD）。对一层 $\mathbf{z}=W\mathbf{x}$，逐样本 Jacobian 是 $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$；一个 batch 的完整 Jacobian 是块对角矩阵，形状 $[B d_{\text{out}},\,B d_{\text{in}}]$。

**形状有多夸张**：$B=32$、$d_{\text{in}}=d_{\text{out}}=4096$ 时，完整 Jacobian 有 $32\times4096\times32\times4096\approx1.7\times10^{10}$ 个元素，bf16 下约 34 GB。**没人会把它存下来。**

**为什么需要**：反向传播只需要「向量–Jacobian 乘积」（vector–Jacobian product, VJP）。给定上游梯度 $\mathbf{v}\in\mathbb{R}^{m}$，它计算 $\mathbf{v}^\top J\in\mathbb{R}^{n}$，成本约等于一次矩阵乘法。对 $\mathbf{z}=W\mathbf{x}$，$\mathbf{v}^\top J=\mathbf{v}^\top W$，形状回到输入空间。**这就是为什么反向传播的显存约等于前向激活，而不是 Jacobian 的尺寸。**

**可检验结论**：对 softmax 输出 $\mathbf{p}\in\mathbb{R}^{V}$（$V=50000$），其 Jacobian 是 $\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$，形状 $[V,V]$，单个样本就有 25 亿元素。但交叉熵损失对 logits 的梯度只是 $\mathbf{p}-\mathbf{y}$，一个 $V$ 维向量——链式法则把巨大的中间矩阵约掉了。

## 9. 练习

1. **概念**：$F:\mathbb{R}^4\to\mathbb{R}^2$ 的 Jacobian 是什么形状？$J_F$ 的第 2 行第 3 列代表什么？
2. **计算**：$F(x,y)=(x+y,\;x^2-y)$，求 $J_F(1,1)$ 与 $\det J_F$。
3. **计算**：$F(x,y,z)=(xy,\;yz)$，求 $J_F$ 在 $(1,2,3)$ 处的值。
4. **链式**：$F:\mathbb{R}^3\to\mathbb{R}^5$，$G:\mathbb{R}^5\to\mathbb{R}^2$，写出 $J_{G\circ F}$ 的形状与乘法顺序。
5. **开放**：为什么深度学习框架选择反向模式而不是前向模式？从「输入维数 vs 输出维数」的角度解释。

<details>
<summary>参考答案</summary>

1. $[2,4]$。第 2 行第 3 列是 $\partial F_2/\partial x_3$：第 2 个输出对第 3 个输入的偏导。
2. $J=\begin{bmatrix}1&1\\2x&-1\end{bmatrix}$，在 $(1,1)$ 处为 $\begin{bmatrix}1&1\\2&-1\end{bmatrix}$，$\det=-1-2=-3$。
3. $J=\begin{bmatrix}y&x&0\\0&z&y\end{bmatrix}$，在 $(1,2,3)$ 处为 $\begin{bmatrix}2&1&0\\0&3&2\end{bmatrix}$，形状 $[2,3]$。
4. $J_G(F(\mathbf{x}))\in\mathbb{R}^{2\times5}$，$J_F\in\mathbb{R}^{5\times3}$，乘积 $J_GJ_F\in\mathbb{R}^{2\times3}$；顺序不能反。
5. 反向模式一次就能得到「标量损失对所有输入」的梯度，成本与前向同阶；前向模式每算一个输入方向就要一遍。大模型输入（参数）上亿、输出（损失）一个标量，所以反向模式压倒性划算。代价是要保存前向激活。

</details>

## 10. 一页速记

- $J_F\in\mathbb{R}^{m\times n}$，$(J_F)_{ij}=\partial F_i/\partial x_j$，行是输出、列是输入。
- $F(\mathbf{x}+\Delta)\approx F(\mathbf{x})+J_F\Delta$，局部线性化。
- $F=A\mathbf{x}+\mathbf{b}$ 的 Jacobian 就是 $A$。
- 链式法则：$J_{G\circ F}=J_GJ_F$，形状 $[p,m][m,n]=[p,n]$。
- 反向传播用 VJP，不显式构造 Jacobian。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-jacobi-1841"></a>**Jacobi（1841）**：*De Determinantibus Functionalibus*。函数行列式（Jacobian）的原始论文；本页把「多元映射的导数」写成矩阵，正是这项工作开启的语言。[Journal für die reine und angewandte Mathematik](https://doi.org/10.1515/crll.1841.22.319)
- <a id="ref-nocedal-wright-2006"></a>**Nocedal & Wright（2006）**：*Numerical Optimization*（2nd ed.）。Jacobian 在非线性方程与优化中的用法，可与本页 §4 的链式法则对照。[Springer](https://doi.org/10.1007/978-0-387-40065-5)
- <a id="ref-goodfellow-2016"></a>**Goodfellow, Bengio & Courville（2016）**：*Deep Learning*。第 6 章用计算图解释 Jacobian 与反向传播，对应本页 §8 的线性层例子。[MIT Press（开放网页版）](https://www.deeplearningbook.org/)

---

下一篇 → [8.4 Hessian 与二阶近似](04-Hessian与二阶近似.md)
