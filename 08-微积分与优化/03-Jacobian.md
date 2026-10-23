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

凌晨 2 点，一次 6 小时的同化窗口要闭合。数值天气预报的模式状态是 **3600 万维**：$600\times300$ 个水平格点、40 层、5 个变量（温度、风的两个分量、湿度、气压）。而这个窗口里进来 **2000 万个观测**：卫星辐射率、探空、飞机报、地面站。

观测算子 $H$ 把 3600 万维的状态映射到 2000 万维的观测空间。业务要的是：**每个观测对初始场的敏感度是多少？把分析增量放在哪几个格点上，才能让 24 小时后的预报误差最小？**

$H$ 的 Jacobian 是 $2000\text{ 万}\times3600\text{ 万}$，$7.2\times10^{14}$ 个数，fp32 要 **2.9 PB**。最快的超算也存不下，更别说在 30 分钟内跑完一次 4D-Var 的 50 次外循环。

所以现实中的做法是：**永远不写这个矩阵，只写它的两个动作**——$J\mathbf{v}$（切线性，把状态扰动变成观测扰动）和 $J^\top\mathbf{w}$（伴随，把观测残差变回状态空间）。代价：如果伴随代码和切线性代码不一致（手写两套，很容易差一个转置），同化会悄悄给出一个偏差方向，24 小时预报的位置误差可能多出几十公里——按一次中等台风估算，就是多疏散两万人。

把现场抽象出来：一个向量值函数的导数，是一个把 $n$ 维扰动翻译成 $m$ 维响应的**线性映射**；选定基之后它就是矩阵。这就是 [Jacobian（雅可比矩阵）](#ref-jacobi-1841)。

读完本节，你应该能回答三件事：

1. Jacobian 的形状为什么是 $[m,n]$，行和列各自代表什么？
2. 链式法则 $J_{G\circ F}=J_GJ_F$ 的顺序由什么决定，写反了会怎样？
3. 为什么反向传播只算向量–Jacobian 乘积，却能得到和显式矩阵一样的结果？

## 1. 直觉

### 1.1 局部线性化（local linearization）

一元函数在一点附近可以用切线近似；多元标量函数用切平面；向量值函数则用**线性映射**近似。

输入空间里一个很小的球，经过 $F$ 后变成一个很小的椭球（或压扁的平行四边形）。Jacobian 就是描述这个「球变椭球」的线性变换矩阵。

### 1.2 每一行是一个梯度

$F$ 的第 $i$ 个输出 $F_i$ 是一个标量函数。它的梯度 $\nabla F_i$ 是行向量（或列向量，取决于约定）。把 $m$ 个输出的梯度摞起来，就是 $m\times n$ 的 Jacobian。

> 💡 **一句话直觉**：**Jacobian = 把每个输出分量的梯度按行叠起来；它是一阶导数的「矩阵版」。**

类比边界：局部线性化只在一点附近有效；离得远时高阶项不可忽略，就像切线不能代替整条曲线。

### 1.3 它在现实世界里对应什么

同化系统里，Jacobian 不是教科书上的方块，而是一张**「观测对格点的敏感度表」**：

| 数学对象 | 同化系统里的对应物 | 单位 |
|---|---|---|
| 输入 $\mathbf{x}$ | 3600 万维模式状态 | K、m/s、kg/kg、Pa |
| 输出 $F(\mathbf{x})$ | 2000 万维模拟观测（辐射率、风、温度） | K（亮温）、m/s 等 |
| $J_{ij}=\partial F_i/\partial x_j$ | 第 $i$ 个观测通道对第 $j$ 个格点变量的敏感度 | 观测单位 / 状态单位 |
| $J\mathbf{v}$ | 给状态加一个扰动 $\mathbf{v}$，观测变多少（切线性） | 观测单位 |
| $J^\top\mathbf{w}$ | 给观测一个残差 $\mathbf{w}$，反推状态该怎么改（伴随） | 状态单位 |

**量级直觉**（经验值，随通道与廓线变化）：一个近地面温度通道对温度的敏感度约 $0.5$ K/K，也就是「状态温度加 1 K，亮温涨 0.5 K」；同一个通道对水汽的敏感度可能在 $10^2$ K/(g/kg) 量级。**同一行里，不同列的单位和量级差好几个数量级**——这就是同化系统必须先对变量做缩放（背景误差协方差的平方根）的原因，否则 Jacobian 的条件数会大到无法迭代。

**边界**：$H$ 含辐射传输，强非线性（云、地表发射率、散射）。$J$ 只是在一条参考廓线附近的线性化；云区的辐射传输甚至不可微（散射相函数切换），这些观测必须做质量控制剔除，否则伴随会放大噪声。**Jacobian 描述的是局部，而大气是一个每小时都在换参考点的系统。**

**这个类比在哪里失效**：矩阵的「行」在数学上没有顺序，但同化里观测是有顺序的（通道 1、通道 2……），$J$ 的行号就是观测索引。如果并行时观测列表被打乱却没有同步重排，矩阵看着还对，物理上已经错位。

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

### 3.1 反事实：把 Jacobian 转置（或把链式顺序写反）会坏在哪

取一个能手算的观测算子，两个输入 $(p,T)$（气压、温度），两个输出：

$$
H(p,T)=\begin{bmatrix}T\\ pT\end{bmatrix}
$$

在 $(p,T)=(2,3)$ 处：

$$
J_H=\begin{bmatrix}0&1\\ T&p\end{bmatrix}=\begin{bmatrix}0&1\\ 3&2\end{bmatrix},\qquad \det J_H=0\cdot2-1\cdot3=-3
$$

现在做一次线性化预报：状态扰动 $\Delta=(0.1,\,0)$（只加气压）。正确的观测变化是

$$
J_H\Delta=\begin{bmatrix}0&1\\3&2\end{bmatrix}\begin{bmatrix}0.1\\0\end{bmatrix}=\begin{bmatrix}0\\0.3\end{bmatrix}
$$

如果代码里错用了转置：

$$
J_H^\top\Delta=\begin{bmatrix}0&3\\1&2\end{bmatrix}\begin{bmatrix}0.1\\0\end{bmatrix}=\begin{bmatrix}0\\0.1\end{bmatrix}
$$

**增量的第二个分量差了 3 倍**。注意两者的行列式都是 $-3$——**在方阵上，转置连行列式都看不出来**，只有真正乘一个扰动量才会暴露。这就是伴随代码必须做点积测试的原因：$\langle J\mathbf{v},\mathbf{w}\rangle=\langle\mathbf{v},J^\top\mathbf{w}\rangle$ 要对随机向量成立到机器精度。

再看链式顺序。设 $F:\mathbb{R}^2\to\mathbb{R}^3$、$G:\mathbb{R}^3\to\mathbb{R}^2$，则 $J_F$ 是 $[3,2]$、$J_G$ 是 $[2,3]$。正确顺序 $J_GJ_F$ 是 $[2,3][3,2]=[2,2]$；写成 $J_FJ_G$ 是 $[3,2][2,3]=[3,3]$——**形状看着也「合理」，但它既不是输入的维数，也不是输出的维数**，是一个物理上不存在的映射。在 4D-Var 里，这等价于把「观测空间」当成了「状态空间」：迭代照样会收敛到某个数，只是那个数没有意义。

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

### 5.4 场景闭环：3600 万维状态、2000 万观测、半小时

**输入**：状态 $n=3.6\times10^7$（fp64 约 288 MB），观测 $m=2\times10^7$（约 160 MB）；背景误差协方差用对角 + 低秩近似，不显式存 $n\times n$。

**计算**：一次 $H(\mathbf{x})$ 要跑 2000 万条辐射传输/插值，按每条 $10^4$ FLOP 估算约 $2\times10^{11}$ FLOP；伴随 $J^\top\mathbf{w}$ 同量级。一轮 4D-Var 里正演与伴随合计要应用约 $2\times10^3$ 次，于是

$$
2\times10^3\times2\times10^{11}=4\times10^{14}\ \text{FLOP}
$$

在 $10^{13}$ FLOP/s 的机器上约 40 秒纯计算。**但实际耗时远不止**——每次应用都要把 288 MB 的状态在几百个节点之间做一次全局转置，通信与 IO 才是瓶颈。

**输出**：分析增量 $\Delta\mathbf{x}\in\mathbb{R}^{3.6\times10^7}$。决策是「这批观测同化还是剔除」「哪个通道的偏差订正系数要更新」。

**成本**：**显式矩阵 $7.2\times10^{14}$ 个数、2.9 PB——不可行**；伴随法只多存两个状态量级的向量（各 288 MB），计算量是前向的 2–3 倍。这就是 Jacobian 在工程里唯一的活法：不存矩阵，只存「乘一下」的代码。

**失败条件与兜底**：

- **伴随与切线性不一致**：最隐蔽的 bug。兜底：每次业务运行前做点积测试，随机取 $\mathbf{v},\mathbf{w}$，要求相对误差 $<10^{-10}$。
- **强非线性区**：台风眼、深对流区，线性化误差超过观测误差。兜底：增加外循环（重线性化），并对这些区域降权。
- **观测偏差未订正**：卫星通道有系统偏差，同化会把偏差吸进状态。兜底：变分偏差订正，把偏差系数和状态一起解。

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

**如果这一步算错，推理/训练时看到什么**：反向传播里漏乘一层转置，梯度形状可能仍然对（方阵时），loss 会下降但明显变慢，验证集早早停在平台；VJP 与 JVP 混用（把 $J^\top\mathbf{v}$ 当成 $J\mathbf{v}$），数值梯度检查立刻对不上，但训练不会报错——它只是朝一个系统性偏斜的方向更新。

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
- **物理意义 / 现实对应**：Jacobian 是一张敏感度表——第 $i$ 行是第 $i$ 个观测对全部状态变量的响应；同化系统只存 $J\mathbf{v}$ 与 $J^\top\mathbf{w}$ 两个动作，从不存矩阵。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-jacobi-1841"></a>**Jacobi（1841）**：*De Determinantibus Functionalibus*。函数行列式（Jacobian）的原始论文；本页把「多元映射的导数」写成矩阵，正是这项工作开启的语言。[Journal für die reine und angewandte Mathematik](https://doi.org/10.1515/crll.1841.22.319)
- <a id="ref-nocedal-wright-2006"></a>**Nocedal & Wright（2006）**：*Numerical Optimization*（2nd ed.）。Jacobian 在非线性方程与优化中的用法，可与本页 §4 的链式法则对照。[Springer](https://doi.org/10.1007/978-0-387-40065-5)
- <a id="ref-goodfellow-2016"></a>**Goodfellow, Bengio & Courville（2016）**：*Deep Learning*。第 6 章用计算图解释 Jacobian 与反向传播，对应本页 §8 的线性层例子。[MIT Press（开放网页版）](https://www.deeplearningbook.org/)

---

下一篇 → [8.4 Hessian 与二阶近似](04-Hessian与二阶近似.md)
