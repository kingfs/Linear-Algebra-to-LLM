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

一个做全球碳汇的团队拿到一张 Web Mercator 的全球栅格：$1440\times720$ 个格子，每格 $0.25^\circ$，标着「森林 / 非森林」。他们要在 2 小时内交出一个数字：格陵兰的冰盖边缘这一年退缩了多少平方公里。

最直接的做法是数格子。格陵兰主体在 $72^\circ$N 附近，覆盖约 9 070 个格子；把每格都按赤道上的 $0.25^\circ\times0.25^\circ\approx772.8$ km² 折算，得到 $9\,070\times772.8=7.01\times10^6$ km²。而格陵兰的真实面积是 $2.166\times10^6$ km²——**高估了 3.2 倍**。

原因不在数据，在投影：Mercator 在纬度 $\varphi$ 处把面积放大 $1/\cos^2\varphi$。$72^\circ$N 时 $\cos72^\circ=0.3090$，$1/\cos^2 72^\circ=10.47$：地面上 238.8 km² 的一格，在图上占 2500 km²。

**决策**：接受「像素计数」只做相对变化（退缩比例），还是给每格乘一个随纬度变化的面积权重、报绝对平方公里？

**代价**：用错权重，局部面积高估 10.47 倍、全球合计高估 3.2 倍；一份按 $7.0\times10^6$ km² 写进报告的碳汇估算会被评审打回，而重跑一次全球栅格统计要 6 小时。

读完本节，你应该能回答三件事：

1. 为什么换元时面积元要乘 $|\det J|$，极坐标的 $r$ 从哪来？
2. 为什么 Jacobian 行列式是逐点的，而线性变换只有一个常数？
3. 投影变形、密度换元、归一化流，用的是同一个公式吗？

## 1. 直觉

### 1.1 局部看，非线性也是线性的

一个非线性映射会把方格网弯成曲线网。但取一个很小的方格，它的像近似是一个平行四边形——因为可微函数在小范围里近似为线性映射。这个平行四边形的面积 ÷ 原方格面积，就是该点的 $|\det J_f|$。所以：

> 💡 **一句话直觉**：**Jacobian 行列式是「逐点的体积缩放因子」；积分时把它乘进面积元。**

### 1.2 一维的类比

一维换元 $\int f(g(x))g'(x)\,dx=\int f(u)\,du$ 里的 $g'(x)$ 就是 $1\times1$ 的 Jacobian。$|g'(x)|$ 告诉你「$x$ 附近的一小段被拉长了几倍」。高维只是把「拉长倍数」换成「体积倍数」。

极坐标的网格是同心圆和射线：半径 $r$ 处的格子，沿 $r$ 方向宽 $dr$，沿 $\theta$ 方向弧长 $r\,d\theta$，面积约 $r\,dr\,d\theta$。那个 $r$ 就是「离原点越远，同样的角宽度对应的弧越长」。类比边界：这个「小格子近似平行四边形」的误差是 $o(\Delta u)$，单独看不为零；但积分取极限后消失。要求 $f$ 连续可微，且 $J_f$ 在积分区域上几乎处处可逆。

### 1.3 它在现实世界里对应什么

地图上，Jacobian 行列式是**面积汇率**：地面上 1 km²，画到这张图上占多少 km²。

| 数学对象 | 地图/气象里的对应物 |
|---|---|
| $J$ | 从经纬度 $(\lambda,\varphi)$ 到平面 $(x,y)$ 的局部线性化 |
| $\lvert\det J\rvert$ | 图上面积 ÷ 地面面积；乘它把地面搬到图上 |
| $\det J$ 随位置变 | 越往高纬，同样的经纬度格子在地图上越大 |
| $\det J=0$ | 投影把区域压成零面积，例如 sinusoidal 投影的极点 |
| $\det J\to\infty$ | Mercator 的极点附近，一格像素代表的地面面积趋于 0 |
| $\operatorname{sign}(\det J)$ | 是否镜像，例如经度方向被翻转的图 |

**量级**：$60^\circ$N 处 $1/\cos^2 60^\circ=4$，地面上 1 km² 在 Mercator 上占 4 km²；$72^\circ$N 处是 10.47；$80^\circ$N 处是 33.2；到 $85^\circ$N 已经是 131.6。Web Mercator 通常截在 $\pm85.05113^\circ$，再往上格子面积趋于无穷。

**这个对应在哪里失效**：Mercator 保角（角度不变）但不保面积，所以「图上量面积」必须加权；换成等面积投影（sinusoidal、EASE-Grid）权重就是常数。另外这是球面到平面的映射，$\det J$ 逐点变化，不能提到积分号外面。

## 2. 定义

### 2.1 Jacobian 矩阵

对 $f:\mathbb{R}^n\to\mathbb{R}^m$，$f(\mathbf{u})=(f_1(\mathbf{u}),\dots,f_m(\mathbf{u}))$，**[Jacobian 矩阵](#ref-jacobi-1841)** 是 $J_f(\mathbf{u})=\big[\partial f_i/\partial u_j\big]_{m\times n}\in\mathbb{R}^{m\times n}$，即第 $i$ 行第 $j$ 列是 $\partial f_i/\partial u_j$。当 $m=n$ 时，$\det J_f$ 有意义，称为 **Jacobian 行列式（Jacobian determinant）**。

### 2.2 变量替换公式（change of variables）

设 $g:U\to\mathbb{R}^n$ 连续可微、在 $U$ 上单射，且 $\det J_g$ 几乎处处不为零，则

$$
\int_{g(U)}f(\mathbf{y})\,d\mathbf{y}
=\int_U f(g(\mathbf{u}))\,|\det J_g(\mathbf{u})|\,d\mathbf{u}.
$$

逐项读法：目标空间的体积元 $d\mathbf{y}$，换成源空间的 $|\det J_g|\,d\mathbf{u}$。绝对值保证体积为正。完整的条件与证明见 [Munkres](#ref-munkres-1991)。

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

### 3.1 反事实：漏掉 $|\det J|$，圆盘面积会变成 $2\pi$

取极坐标换元，但假装 $dx\,dy=dr\,d\theta$（把 $|\det J|$ 当成 1）。单位圆盘的面积变成

$$
\int_0^{2\pi}\int_0^1 1\,dr\,d\theta=2\pi,
$$

而真实面积是 $\pi$。**算出来的正好是半径为 $\sqrt2$ 的圆盘**——同一个圆盘，只因为换了坐标就变大一倍，而且没有任何数值误差，全是精确算术。

再取 $f(u,v)=(u+v,\,u-v)$，$|\det J|=2$。如果写成 $d\mathbf{y}=\det J\,d\mathbf{u}=-2\,d\mathbf{u}$ 而不取绝对值，一个真实面积为 2 的菱形会得到 $-2$——负面积。

这两种错法都能通过「量纲检查」（结果都是面积），却分别错在**漏了局部比例**和**漏了绝对值**。公式里 $|\det J|$ 的每一部分都不是装饰。

对地图同理：把每个 $0.25^\circ$ 格子的面积都当成赤道上的 772.8 km²，相当于漏掉 $\cos\varphi$ 这个局部因子，于是格陵兰那 9 070 个格子被算成 $7.01\times10^6$ km²——正是第 0 节高估 3.2 倍的结果。

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

### 5.5 场景闭环：1 036 800 个格子，20 毫秒

**输入**：$1440\times720=1\,036\,800$ 个 $0.25^\circ$ 格子。每个格子的面积权重用 float32 存，一张图 $1\,036\,800\times4\ \text{B}\approx4.1$ MB；森林比例另存一张。

**计算**：每格的真实面积是 $R^2\Delta\lambda\Delta\varphi\cos\varphi$，$R=6371$ km、$\Delta\lambda=\Delta\varphi=0.25^\circ=0.004363$ rad：

$$
R^2\Delta\lambda\Delta\varphi=772.8\ \text{km}^2,\qquad A(\varphi)=772.8\cos\varphi.
$$

$72^\circ$N 处 $A=238.8$ km²，和 §0 的数对得上：$2.166\times10^6/238.8=9\,070$。对 100 万个格子各算一次 $\cos\varphi$，单核约 20 ms。

**规模感**：这里每个点是一个 $2\times2$ 行列式，3 次乘法；100 万个格子合计约 $3\times10^6$ 次乘法，几毫秒。换成 100 维的换元（flow 的一层），每个点的 $J$ 是 $100\times100$，一次 $\det$ 要 $\frac23 n^3\approx6.7\times10^5$ 次乘加；同样 100 万个点就是 $6.7\times10^{11}$ 次，单核 1 GFLOP/s 要 670 秒。**这就是 coupling 层把 $\det$ 变成对角元乘积、把 $O(D^3)$ 降到 $O(D)$ 的现实动机。**

**输出**：加权面积 $2.17\times10^6$ km²，对比像素计数的 $7.01\times10^6$ km²。决策：绝对面积必须加权；相对变化可以用像素比，前提是两期用同一张投影。

**成本**：20 ms CPU、4.1 MB 内存，相比 6 小时的人工复核可以忽略。真正的成本是**验证**：所有权重之和应等于 $4\pi R^2=5.10\times10^8$ km²，差一点就说明经纬度网格或 $\cos\varphi$ 写错了。

**失败条件**：

- **极点权重**：$\varphi\to90^\circ$ 时 $\cos\varphi\to0$，权重趋于 0，而 Mercator 的 $y$ 趋于无穷；Web Mercator 截在 $\pm85.05113^\circ$，两极的格子直接缺失。兜底：换等面积投影，或在球面网格上积分。
- **经度环绕**：跨 $180^\circ$ 经线的格子被切成两半，按像素数会重复计数。兜底：在球面坐标里处理边界。
- **分辨率不一致**：两期栅格一个 $0.25^\circ$、一个 $0.1^\circ$，像素比没有意义。兜底：重采样到同一网格再比。

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

**怎么算得动**：一般 $D\times D$ 矩阵的 $\det$ 是 $O(D^3)$，且高维容易下溢。Coupling 层（[Dinh 等](#ref-dinh-2016)）把 $z$ 分成两半：

$$
x_1=z_1,\qquad x_2=z_2\odot s(z_1)+t(z_1).
$$

它的 Jacobian 是块三角阵，$\det J=\prod_i s_i(z_1)$，于是 $\log|\det J|=\sum_i\log|s_i(z_1)|$，只要 $O(D)$。这是 flow 能堆很多层的关键。

**数值坑**：$D=1024$、每维缩放 $0.1$ 时，$\det J=10^{-1024}$，float64 直接下溢为 0，$\log$ 变 $-\infty$；每维缩放 $10$ 则上溢。必须在对数域累加，或调用 `slogdet`。**可检验结论**：`np.linalg.det` 对这种矩阵返回 0.0，而 `np.linalg.slogdet` 仍能给出约 $-2357.8$ 的有限值。

**算错时下游看到什么**：漏掉 $\log|\det J|$，密度积分不再等于 1，模型可以靠缩小体积把 $\log p_X$ 抬到任意大——训练 loss 一路下降，但采样出的分布与数据无关；忘记绝对值，遇到 $\det J<0$ 的层时 $\log$ 负数，loss 一次前向就变 NaN。两种故障都不是「精度差一点」，而是目标函数本身错了。

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
- **物理意义 / 现实对应**：$|\det J|$ 是面积汇率——地面上 1 km² 在 $72^\circ$N 的 Mercator 上占 10.47 km²。
- 变量替换：$d\mathbf{y}=|\det J_g(\mathbf{u})|\,d\mathbf{u}$；概率密度要除以它。
- 极坐标 $\det J=r$，所以 $dx\,dy=r\,dr\,d\theta$；球坐标 $\det J=\rho^2\sin\varphi$。
- 链式法则让 $\det$ 相乘，$\log|\det|$ 相加；三角 $J$ 的 $\det$ 是对角元乘积。
- 高维用 $\log|\det|$（`slogdet`），别直接算 $\det$。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-jacobi-1841"></a>**Jacobi（Carl Gustav Jacob Jacobi, 1841）**：*De determinantibus functionalibus*。把 $n$ 个函数对 $n$ 个变量的偏导行列式作为独立对象研究，并给出复合时的乘法规则；「Jacobian 行列式」的名字由此而来，本页第 2.1 节与第 4 节的链式法则都建立在这个对象上。[Journal für die reine und angewandte Mathematik](https://doi.org/10.1515/crll.1841.22.319)
- <a id="ref-munkres-1991"></a>**Munkres（James R. Munkres, 1991；CRC 2018 重印）**：*Analysis on Manifolds*。给出变量替换定理的完整陈述与证明，包括 $C^1$、单射、$\det J_g$ 几乎处处非零这些条件各自的作用——本页第 2.2 节的公式以它为准。[CRC Press](https://doi.org/10.1201/9780429494147)
- <a id="ref-dinh-2016"></a>**Dinh 等（Laurent Dinh, Jascha Sohl-Dickstein & Samy Bengio, 2016）**：*Density estimation using Real NVP*。提出仿射 coupling 层 $x_2=z_2\odot s(z_1)+t(z_1)$，其块三角 Jacobian 让 $\log|\det J|$ 只需 $O(D)$；本页第 8 节的设计直接取自这篇。[arXiv:1605.08803](https://arxiv.org/abs/1605.08803)

---

本篇是模块 5 的最后一篇。回到 [模块导览](README.md) 做自测，或去做 [习题](../exercises/m05-习题.md)。
