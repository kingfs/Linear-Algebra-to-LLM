---
id: 5.4
title: 行列式与 Jacobian：变量替换
module: M05
slug: determinant-jacobian
prereq: [5.1, 8.3]
minutes: 35
level: 进阶
tags: [Jacobian, 变量替换, 极坐标, 归一化流]
llm_hooks: [change-of-variables, normalizing-flow, log-determinant]
---

# 5.4 行列式与 Jacobian：变量替换

> **本节位置**：模块 5 · 行列式 ｜ **前置**：[5.1](01-行列式几何.md)、[8.3 Jacobian](../08-微积分与优化/03-Jacobian.md) ｜ **预计**：35 分钟
>
> **一句话**：非线性变换没有全局的 $\det$，但在每一点有一个局部体积缩放因子——那就是 Jacobian 行列式。

## 0. 场景引入

要算单位圆盘上的积分 $\iint_D (x^2+y^2)\,dx\,dy$。直角坐标下要分上下半圆，很麻烦；换成极坐标（polar coordinates）$x=r\cos\theta,\ y=r\sin\theta$ 后，积分变成 $\int_0^{2\pi}\int_0^1 r^2\cdot r\,dr\,d\theta$。注意最后多出来的那个 $r$：$dx\,dy$ 不是变成 $dr\,d\theta$，而是变成 $r\,dr\,d\theta$。

**问题**：这个 $r$ 从哪来？为什么不是 1？如果换一个一般的变换 $x=f(u,v)$，面积元又该怎么换？

## 1. 直觉

### 1.1 局部看，非线性也是线性的

一个非线性映射会把方格网弯成曲线网。但取一个很小的方格，它的像近似是一个平行四边形——因为可微函数在小范围里近似为线性映射。这个平行四边形的面积 ÷ 原方格面积，就是该点的 $|\det J_f|$。所以：

> 💡 **一句话直觉**：**Jacobian 行列式是「逐点的体积缩放因子」；积分时把它乘进面积元。**

### 1.2 一维的类比

一维换元 $\int f(g(x))g'(x)\,dx=\int f(u)\,du$ 里的 $g'(x)$ 就是 $1\times1$ 的 Jacobian。$|g'(x)|$ 告诉你「$x$ 附近的一小段被拉长了几倍」。高维只是把「拉长倍数」换成「体积倍数」。

极坐标的网格是同心圆和射线：半径 $r$ 处的格子，沿 $r$ 方向宽 $dr$，沿 $\theta$ 方向弧长 $r\,d\theta$，面积约 $r\,dr\,d\theta$。那个 $r$ 就是「离原点越远，同样的角宽度对应的弧越长」。类比边界：这个「小格子近似平行四边形」的误差是 $o(\Delta u)$，单独看不为零；但积分取极限后消失。要求 $f$ 连续可微，且 $J_f$ 在积分区域上几乎处处可逆。

## 2. 定义

### 2.1 Jacobian 矩阵

对 $f:\mathbb{R}^n\to\mathbb{R}^m$，$f(\mathbf{u})=(f_1(\mathbf{u}),\dots,f_m(\mathbf{u}))$，**Jacobian 矩阵**是 $J_f(\mathbf{u})=\big[\partial f_i/\partial u_j\big]_{m\times n}\in\mathbb{R}^{m\times n}$，即第 $i$ 行第 $j$ 列是 $\partial f_i/\partial u_j$。当 $m=n$ 时，$\det J_f$ 有意义，称为 **Jacobian 行列式（Jacobian determinant）**。

### 2.2 变量替换公式（change of variables）

设 $g:U\to\mathbb{R}^n$ 连续可微、在 $U$ 上单射，且 $\det J_g$ 几乎处处不为零，则

$$
\int_{g(U)}f(\mathbf{y})\,d\mathbf{y}
=\int_U f(g(\mathbf{u}))\,|\det J_g(\mathbf{u})|\,d\mathbf{u}.
$$

逐项读法：目标空间的体积元 $d\mathbf{y}$，换成源空间的 $|\det J_g|\,d\mathbf{u}$。绝对值保证体积为正。

### 2.3 概率密度（probability density）变换

若 $Y=g(X)$ 且 $g$ 可逆，则

$$
p_Y(\mathbf{y})=\frac{p_X(g^{-1}(\mathbf{y}))}{|\det J_g(g^{-1}(\mathbf{y}))|}.
$$

线性情形 $g(\mathbf{u})=A\mathbf{u}$ 时 $J_g=A$，退化为 5.1 的 $p_Y(\mathbf{y})=p_X(A^{-1}\mathbf{y})/|\det A|$。

## 3. 为什么这样定义

把源区域切成很多边长 $\Delta u$ 的小方块。第 $k$ 个小方块的体积是 $\Delta u^n$；它的像近似是平行多面体，体积是 $|\det J_g(\mathbf{u}_k)|\Delta u^n$。

于是 $\sum_k f(g(\mathbf{u}_k))\,|\det J_g(\mathbf{u}_k)|\,\Delta u^n\longrightarrow \int_U f(g(\mathbf{u}))\,|\det J_g(\mathbf{u})|\,d\mathbf{u}$，右边正是目标空间里对 $f$ 的黎曼和。这就是变量替换公式的全部内容：**先把体积按局部比例换算，再求和**。

导数的链式法则保证：先做 $g$ 再做 $h$，总的体积因子是 $|\det J_h|\cdot|\det J_g|$，与「逐点缩放相乘」一致。

## 4. 性质与推导

- **链式法则**：$J_{h\circ g}(\mathbf{u})=J_h(g(\mathbf{u}))J_g(\mathbf{u})$，因此 $\det J_{h\circ g}=\det J_h\cdot\det J_g$。
- **逆映射**：$\det J_{g^{-1}}(\mathbf{y})=1/\det J_g(\mathbf{u})$，其中 $\mathbf{y}=g(\mathbf{u})$。
- **线性映射**：$J_A=A$，$\det J_A=\det A$，回到 5.1。
- **三角 Jacobian**：若 $J$ 是三角阵，$\det J=\prod_i J_{ii}$；归一化流的 coupling 层就靠这一点把 $O(D^3)$ 降到 $O(D)$。
- **极坐标**：$\det J=r$；球坐标：$\det J=\rho^2\sin\varphi$。
- **定向**：$\det J<0$ 表示局部翻转；积分用 $|\det J|$，所以面积/体积仍为正。

## 5. 例子

### 5.1 极坐标：完整的推导

$$
x=r\cos\theta,\quad y=r\sin\theta,\qquad
J=\begin{bmatrix}\cos\theta&-r\sin\theta\\ \sin\theta&r\cos\theta\end{bmatrix}.
$$

$$
\det J=\cos\theta\cdot r\cos\theta-(-r\sin\theta)\cdot\sin\theta
=r(\cos^2\theta+\sin^2\theta)=r.
$$

所以 $dx\,dy=r\,dr\,d\theta$。单位圆盘面积：

$$
\int_0^{2\pi}\int_0^1 r\,dr\,d\theta=2\pi\cdot\frac12=\pi.
$$

高斯积分也变简单：$\iint e^{-(x^2+y^2)}dx\,dy=\int_0^{2\pi}\int_0^\infty e^{-r^2}r\,dr\,d\theta=2\pi\cdot\frac12=\pi$。

### 5.2 线性变换：面积比是 $|\det|$

$f(u,v)=(u+v,\,u-v)$，$J=\begin{bmatrix}1&1\\1&-1\end{bmatrix}$，$\det J=-2$。

单位正方形的四个顶点映到 $(0,0)$、$(1,1)$、$(2,0)$、$(1,-1)$，围成菱形，面积 $=2=|\det J|$。$\det<0$ 说明定向翻转，但面积仍取绝对值。

### 5.3 非线性变换：缩放逐点变化

$f(u,v)=(u^2,\,v)$，$J=\begin{bmatrix}2u&0\\0&1\end{bmatrix}$，$\det J=2u$。

在 $u=0.5$ 处局部面积因子为 1，在 $u=2$ 处为 4。这不是一个全局常数，必须放在积分号里面。

对单位正方形：$\int_0^1\int_0^1 2u\,du\,dv=1$，而像区域是 $[0,1]\times[0,1]$，面积确实为 1——局部缩放有大有小，积分后正好抵消。

### 5.4 概率：线性变换后的密度

设 $X\sim\mathcal{N}(\mathbf{0},I_2)$，$Y=AX$，$A=\operatorname{diag}(2,3)$，$\det A=6$，则 $p_Y(\mathbf{y})=\frac{1}{6}\cdot\frac{1}{2\pi}\exp\!\left(-\frac12\|A^{-1}\mathbf{y}\|^2\right)$。在原点：$p_Y(\mathbf{0})=1/(12\pi)\approx0.0265$。可以验证：$Y$ 的协方差是 $AA^\top=\operatorname{diag}(4,9)$，二维高斯在原点处的密度正是 $1/(2\pi\sqrt{36})=1/(12\pi)$。

## 6. 常见误区

1. **误区**：$dx\,dy=dr\,d\theta$。
   **纠正**：极坐标下是 $r\,dr\,d\theta$，漏掉 $r$ 是最经典的错误。$r=2$ 处的角宽度 $d\theta$ 对应的弧长是 $2\,d\theta$，面积自然多一个因子 2。

2. **误区**：$\det J$ 是一个常数，可以提到积分号外。
   **纠正**：只有线性变换才是常数。非线性时 $J$ 逐点变化，必须留在积分号内。

3. **误区**：换元时用 $g^{-1}$ 的 Jacobian 还是 $g$ 的，随便。
   **纠正**：记住「源空间体积元 × $|\det J_g|$ = 目标空间体积元」。若用 $p_Y$ 的公式，分母是 $g$ 的 Jacobian 在 $g^{-1}(y)$ 处的值。

4. **误区**：$J$ 在某点奇异，换元就失效。
   **纠正**：失效只发生在该点附近；如果奇异点集是零测集（如极坐标的 $r=0$），积分仍成立。

## 7. 概念网络

- 前置：[5.1 行列式几何](01-行列式几何.md) —— 线性变换的体积缩放，是这里的局部版本。
- 前置：[5.2 行列式的性质与计算](02-行列式的性质与计算.md) —— 算 $2\times2$、$3\times3$ 的 $\det J$。
- 关联：[5.3 逆与 Cramer 法则](03-逆与Cramer法则.md) —— 逆映射的 Jacobian 是原 Jacobian 的逆。
- 后续：[8.3 Jacobian](../08-微积分与优化/03-Jacobian.md) —— 把这里的矩阵当作多元导数，反向传播是它的链式法则。
- 后续：9.2 多元高斯—— 密度变换与协方差行列式。
- 关联：[1.7 叉积](../01-向量与几何/07-叉积.md) —— 二维面积与三维平行六面体体积。

## 8. 通往大模型：归一化流的 $\log|\det J|$

**出现位置**：归一化流（normalizing flow）的似然计算、概率密度变换、可逆残差网络。

**形状**：$z\in\mathbb{R}^D$，可逆网络 $x=f(z)$，$J_f=\partial f/\partial z\in\mathbb{R}^{D\times D}$。训练时要算

$$
\log p_X(x)=\log p_Z(z)-\log|\det J_f(z)|.
$$

**为什么需要**：生成模型要最大化数据似然。密度变换必须带体积修正；少了这一项，优化的就不再是数据空间的归一化似然：模型可以靠任意改变体积把密度数值抬高，而不用为体积变化负责（偏好放大还是缩小，取决于网络参数化成 $z\to x$ 还是 $x\to z$）。

**怎么算得动**：一般 $D\times D$ 矩阵的 $\det$ 是 $O(D^3)$，且高维容易下溢。Coupling 层把 $z$ 分成两半：

$$
x_1=z_1,\qquad x_2=z_2\odot s(z_1)+t(z_1).
$$

它的 Jacobian 是块三角阵，$\det J=\prod_i s_i(z_1)$，于是 $\log|\det J|=\sum_i\log|s_i(z_1)|$，只要 $O(D)$。这是 flow 能堆很多层的关键。

**数值坑**：$D=1024$、每维缩放 $0.1$ 时，$\det J=10^{-1024}$，float64 直接下溢为 0，$\log$ 变 $-\infty$；每维缩放 $10$ 则上溢。必须在对数域累加，或调用 `slogdet`。**可检验结论**：`np.linalg.det` 对这种矩阵返回 0.0，而 `np.linalg.slogdet` 仍能给出约 $-2357.8$ 的有限值。

## 9. 练习

1. **推导**：用 $x=r\cos\theta,\ y=r\sin\theta$ 推导 $\det J=r$，并解释为什么是 $r$ 而不是 1。
2. **计算**：$f(u,v)=(u+v,\,u-v)$。求 $J$、$\det J$，以及单位正方形像的面积；定向翻转了吗？
3. **计算**：$f(x,y)=(e^x\cos y,\,e^x\sin y)$。求 $J$ 与 $\det J$；在 $(0,\pi/4)$ 处局部面积放大几倍？
4. **概率**：$X\sim\mathcal{N}(\mathbf{0},I_2)$，$Y=AX$，$A=\operatorname{diag}(2,3)$。求 $p_Y(\mathbf{0})$，并与 $p_X(\mathbf{0})$ 比较。
5. **开放**：归一化流为什么偏爱「三角 Jacobian」的 coupling 层？它在表达能力上牺牲了什么？

<details>
<summary>参考答案</summary>

1. $J=\begin{bmatrix}\cos\theta&-r\sin\theta\\ \sin\theta&r\cos\theta\end{bmatrix}$，$\det J=r\cos^2\theta+r\sin^2\theta=r$。半径 $r$ 越大，同样的 $d\theta$ 对应的弧长越长（$r\,d\theta$），面积元多出因子 $r$。
2. $J=\begin{bmatrix}1&1\\1&-1\end{bmatrix}$，$\det=-2$；像是顶点 $(0,0),(1,1),(2,0),(1,-1)$ 的菱形，面积 $2=|\det|$；$\det<0$，定向翻转。
3. $J=\begin{bmatrix}e^x\cos y&-e^x\sin y\\ e^x\sin y&e^x\cos y\end{bmatrix}$，$\det=e^{2x}(\cos^2y+\sin^2y)=e^{2x}$；在 $(0,\pi/4)$ 处为 $e^0=1$，局部保面积。
4. $p_X(\mathbf{0})=1/(2\pi)$；$p_Y(\mathbf{0})=p_X(\mathbf{0})/|\det A|=1/(2\pi\cdot6)=1/(12\pi)\approx0.0265$。
5. 三角 Jacobian 让 $\det$ 变成对角元乘积、$\log|\det|$ 变成 $O(D)$ 的求和，数值稳定且反向传播便宜。代价是每层只变换一部分维度（另一半保持不变），需要堆叠多层并交替划分，单层表达能力受限。

</details>

## 10. 一页速记

- 非线性映射的局部体积因子是 $|\det J_f(\mathbf{u})|$，逐点变化。
- 变量替换：$d\mathbf{y}=|\det J_g(\mathbf{u})|\,d\mathbf{u}$；概率密度要除以它。
- 极坐标 $\det J=r$，所以 $dx\,dy=r\,dr\,d\theta$；球坐标 $\det J=\rho^2\sin\varphi$。
- 链式法则让 $\det$ 相乘，$\log|\det|$ 相加；三角 $J$ 的 $\det$ 是对角元乘积。
- 高维用 $\log|\det|$（`slogdet`），别直接算 $\det$。

---

本篇是模块 5 的最后一篇。回到 [模块导览](README.md) 做自测，或去做 [习题](../exercises/m05-习题.md)。
