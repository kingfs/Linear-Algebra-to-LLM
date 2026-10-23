---
id: 8.4
title: Hessian 与二阶近似
module: M08
slug: hessian
prereq: [6.5, 8.3]
minutes: 40
level: 核心
tags: [Hessian, 二阶近似, 曲率, 正定]
llm_hooks: [second-order-optimization, loss-landscape]
---

# 8.4 Hessian 与二阶近似

> **本节位置**：模块 8 · 微积分与优化 ｜ **前置**：6.5 二次型、[8.3 Jacobian](03-Jacobian.md) ｜ **预计**：40 分钟
>
> **一句话**：Hessian 是二阶偏导组成的对称矩阵，它告诉你函数在一点是碗、是帽、还是马鞍，以及各个方向有多弯。

## 0. 场景引入

一座 40 m 跨径的钢桁架要减重 **30%**。拓扑优化把设计域切成 **12 万个**有限元单元，每个单元一个密度变量 $\rho_e\in[0,1]$，目标是最小化柔度

$$
c(\boldsymbol\rho)=\mathbf{f}^\top\mathbf{u},\qquad K(\boldsymbol\rho)\,\mathbf{u}=\mathbf{f}
$$

柔度的单位是焦耳：载荷做的功越小，结构越刚。梯度能算——伴随法一次求解就得到 12 万个数。但二阶信息呢？

Hessian 是 $12\text{ 万}\times12\text{ 万}$，$1.44\times10^{10}$ 个数，fp64 要 **115 GB**——比一个计算节点的内存还大。一次 Cholesky 分解约 $n^3/3\approx5.8\times10^{14}$ FLOP，在 100 GFLOP/s 上要 **1.6 小时**，而你总共只有 60 次迭代的预算。

**决策**：到底要不要二阶信息？如果要，在连矩阵都存不下的情况下，怎么用它？

**代价**：盲目上牛顿法，某一步 Hessian 不定，方向指向上升，柔度反而变大；密度被推到 0/1 边界后出现棋盘格，验证不通过，重做一轮又是 3 天。

这不是学术问题。同样的矩阵出现在 70 亿维的损失曲面上（§8）：初始化落在马鞍附近，梯度很小，训练长时间停滞；某个方向曲率极大，学习率稍大就震荡。Hessian 就是描述这些的矩阵。

读完本节，你应该能回答三件事：

1. Hessian 的特征值如何决定一个驻点是碗、是帽还是马鞍？
2. 为什么 $n=12$ 万时 Hessian 存不下，而我们仍然能用它？
3. 「Hessian–向量积」凭什么只要约 2 次反向传播的代价？

## 1. 直觉

### 1.1 从「斜率」到「斜率的变化」

一元函数 $f(x)$ 在 $x_0$ 附近有 $f(x_0+\Delta)\approx f(x_0)+f'(x_0)\Delta+\tfrac12 f''(x_0)\Delta^2$。$f''$ 决定抛物线开口朝上还是朝下。多元时，每个方向都有自己的「开口」，Hessian 把所有方向的二阶信息打包。

### 1.2 碗、帽与马鞍

- $H$ 正定：所有方向都向上弯 → 碗，局部极小；
- $H$ 负定：所有方向都向下弯 → 帽，局部极大；
- $H$ 不定：有的向上、有的向下 → 马鞍。

特征值就是「主轴方向上的弯曲程度」，也就是主曲率（principal curvature），特征向量就是这些主轴。

> 💡 **一句话直觉**：**Hessian 是曲率的矩阵；特征值告诉你每个方向多弯，特征向量告诉你朝哪弯。**

### 1.3 它在现实世界里对应什么

在拓扑优化里，Hessian 有一个非常具体的身份：**刚度的二阶敏感度**。

| 数学对象 | 桁架优化里的对应物 | 单位 |
|---|---|---|
| $\rho_e$ | 第 $e$ 个单元的密度（0 = 挖空，1 = 实心） | 无量纲 |
| $c(\boldsymbol\rho)$ | 柔度 $\mathbf{f}^\top\mathbf{u}$ | J |
| $\nabla c$ | 每个单元密度加一点，柔度变多少 | J |
| $H_{ef}=\partial^2c/\partial\rho_e\partial\rho_f$ | 单元 $e$ 与 $f$ 之间的刚度耦合 | J |
| $H$ 的特征值 | 沿某个密度组合方向，刚度曲线的弯曲程度 | J |

因为密度无量纲，$H$ 的单位就是焦耳，和 $c$ 一样。**如果变量换成有量纲的量**（比如节点位移，单位 m），$H$ 的单位就变成 J/m²——数值跟着单位变，这一点和 8.1 节的梯度完全一致。

**量级直觉**（经验值）：$\lambda_{\max}$ 常在 $10^3$ J 量级；而接近机构学奇异（某段结构快要变成可动机构）时，$\lambda_{\min}$ 会掉到 $10^{-3}$ 甚至更低，条件数 $10^6$。**这不是「有一点病态」，而是「牛顿方向基本不可信」**——这正是工程上宁可用 MMA/OC 这类一阶方法的原因。

**边界**：柔度对密度**不是凸的**（SIMP 惩罚 $p=3$ 让中间密度被惩罚），所以 $H$ 在迭代中会变号。Hessian 只描述当前点的局部曲率；密度接近 0/1 时，$c$ 对 $\rho$ 的导数趋于奇异，二阶展开迅速失效。

**这个类比在哪里失效**：结构力学里的刚度矩阵 $K$ 是正定的（稳定结构），但 $c$ 的 Hessian **不是** $K$——它含 $K^{-1}$ 与 $\partial K/\partial\rho$ 的乘积，完全可能不定。把「刚度矩阵正定」的直觉套到 Hessian 上，是这一节最常见的错误。

## 2. 定义

设 $f:\mathbb{R}^n\to\mathbb{R}$ 二阶连续可微，Hessian（黑塞矩阵）定义为

$$
H_f(\mathbf{x})=
\begin{bmatrix}
\dfrac{\partial^2 f}{\partial x_1^2}&\cdots&\dfrac{\partial^2 f}{\partial x_1\partial x_n}\\
\vdots&\ddots&\vdots\\
\dfrac{\partial^2 f}{\partial x_n\partial x_1}&\cdots&\dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}\in\mathbb{R}^{n\times n}
$$

即 $(H_f)_{ij}=\partial^2 f/\partial x_i\partial x_j$。

**对称性**：若二阶偏导连续，则 $\partial^2 f/\partial x_i\partial x_j=\partial^2 f/\partial x_j\partial x_i$（Clairaut/Schwarz 定理），所以 $H$ 对称。对称矩阵因此有实特征值和正交特征向量（6.4 节）。

**与 Jacobian 的关系**：$H_f=J_{\nabla f}$。梯度是向量值函数，它的 Jacobian 就是 Hessian。这也解释了为什么 $H$ 的行列都对应输入维度。

## 3. 为什么这样定义

一阶导数给出「往哪走」，但不给出「走一步之后斜率怎么变」。没有二阶信息：

- 无法判断驻点是极小、极大还是鞍点；
- 无法估计学习率的安全上界；
- 牛顿法无从谈起（它要用 $H^{-1}$ 修正方向）。

为什么是矩阵而不是一组数？因为不同方向的弯曲可以不同，而且方向之间可能「耦合」——沿 $x$ 走时，$y$ 方向的斜率也会变，这个耦合就写在非对角元 $H_{xy}$ 里。

### 3.1 反事实：Hessian 不定还用牛顿法，会朝鞍点跑

取 $f(x,y)=x^2-y^2$，在 $(0.1,\,1)$ 处：

$$
\nabla f=\begin{bmatrix}2x\\-2y\end{bmatrix}=\begin{bmatrix}0.2\\-2\end{bmatrix},\qquad
H=\begin{bmatrix}2&0\\0&-2\end{bmatrix}
$$

$H$ 的特征值是 $2$ 与 $-2$，不定。牛顿步：

$$
\mathbf{x}_1=\mathbf{x}_0-H^{-1}\nabla f
=\begin{bmatrix}0.1\\1\end{bmatrix}-\begin{bmatrix}0.1\\1\end{bmatrix}
=\begin{bmatrix}0\\0\end{bmatrix}
$$

函数值从 $f(0.1,1)=0.01-1=-0.99$ 变成 $f(0,0)=0$——**上升了 0.99**，而且正好落在鞍点上。牛顿法用 $-2$ 这个负曲率把 $y$ 方向「翻转」了：它以为自己在往谷底走，实际上在往山顶走。

对照梯度下降，$\eta=0.1$：

$$
\mathbf{x}_1=\begin{bmatrix}0.1\\1\end{bmatrix}-0.1\begin{bmatrix}0.2\\-2\end{bmatrix}
=\begin{bmatrix}0.08\\1.2\end{bmatrix},\qquad f=0.0064-1.44=-1.4336<-0.99
$$

**同一个点，牛顿法上升，梯度下降下降。** 所以实用牛顿法都要加保护：Hessian 不定时改用 $H+\tau I$（$\tau$ 取到最小特征值为正），或者干脆沿负梯度做线搜索（信赖域 / Levenberg–Marquardt）。

顺带一个反事实：如果只检查「$H$ 的行列式」，$2\times(-2)=-4<0$ 确实能发现不定；但 $3\times3$ 以上，行列式为正也可能是「两正一负」的不定矩阵。**必须看特征值（或顺序主子式），不能只看行列式。**

## 4. 性质与推导

### 4.1 二阶 Taylor 展开

$$
f(\mathbf{x}+\Delta)=f(\mathbf{x})+\nabla f(\mathbf{x})^\top\Delta
+\tfrac12\Delta^\top H_f(\mathbf{x})\Delta+o(\|\Delta\|^2)
$$

逐项读法：当前值 + 一阶（切平面）+ 二阶（抛物面）。二次函数的高阶项为零，所以这个展开对二次函数是**精确**的。

### 4.2 二阶判据

设 $\nabla f(\mathbf{x}^\star)=\mathbf{0}$：

| $H_f(\mathbf{x}^\star)$ | 结论 |
|---|---|
| 正定（positive definite，所有特征值 $>0$） | 严格局部极小 |
| 负定（所有特征值 $<0$） | 严格局部极大 |
| 不定（有正有负） | 鞍点 |
| 半正定/半负定（含 0 特征值） | 判据失效，需看高阶项 |

正定的定义是：对任意 $\Delta\neq\mathbf{0}$，$\Delta^\top H\Delta>0$。在驻点处，二阶项是主导项，所以它为正意味着往任何方向走函数都变大。

### 4.3 二次型与最小二乘

对 $f(\mathbf{x})=\frac12\mathbf{x}^\top A\mathbf{x}$（$A$ 对称）：$\nabla f=A\mathbf{x}$，$H_f=A$。对最小二乘 $f(\boldsymbol\theta)=\frac12\|X\boldsymbol\theta-\mathbf{y}\|^2$：$\nabla f=X^\top(X\boldsymbol\theta-\mathbf{y})$，$H_f=X^\top X$。

$X^\top X$ 半正定；当 $X$ 列满秩时正定。它的特征值是 $X$ 的奇异值平方（7.2 节），条件数 $\kappa=\sigma_{\max}^2/\sigma_{\min}^2$ 决定优化难度。

### 4.4 特征分解与坐标解耦

$H=Q\Lambda Q^\top$。令 $\Delta=Q\mathbf{c}$，二阶项变成 $\tfrac12\Delta^\top H\Delta=\tfrac12\sum_i\lambda_i c_i^2$。读法：在特征坐标下，曲面是一堆独立抛物线的和，每个方向的曲率就是 $\lambda_i$。$\lambda_{\max}/\lambda_{\min}$ 越大，曲面越「狭长」，优化越难。

## 5. 例子

### 5.1 手算：判定鞍点

设 $f(x,y)=x^2+3xy+y^2$，梯度 $\nabla f=(2x+3y,\;3x+2y)^\top$。令其为零，得 $x=y=0$。Hessian：

$$
H=\begin{bmatrix}2&3\\3&2\end{bmatrix}
$$

迹 $=4$，行列式 $=4-9=-5$。特征值满足 $\lambda^2-4\lambda-5=0$，得 $\lambda_1=5,\lambda_2=-1$。

一正一负 → 不定 → $(0,0)$ 是**鞍点**。沿特征向量 $(1,1)/\sqrt2$ 曲率为 5（向上弯），沿 $(1,-1)/\sqrt2$ 曲率为 $-1$（向下弯）。

### 5.2 手算：二阶 Taylor 精确到小数点后

设 $f(x,y)=x^2+2y^2$，$\mathbf{x}_0=(1,1)$，$\Delta=(0.1,-0.1)^\top$。$f(\mathbf{x}_0)=3$，$\nabla f=(2,4)^\top$，$H=\operatorname{diag}(2,4)$。

$$
f\approx 3+(2,4)\begin{bmatrix}0.1\\-0.1\end{bmatrix}
+\frac12(0.1,-0.1)\begin{bmatrix}2&0\\0&4\end{bmatrix}\begin{bmatrix}0.1\\-0.1\end{bmatrix}
=3-0.2+\frac12(0.02+0.04)=2.83
$$

精确值：$1.1^2+2\times0.9^2=1.21+1.62=2.83$。因为 $f$ 是二次函数，二阶展开**没有误差**。

### 5.3 场景：正定二次型与条件数

设 $f(x,y)=x^2+xy+y^2$，$H=\begin{bmatrix}2&1\\1&2\end{bmatrix}$。特征值 $1$ 和 $3$，都为正，所以原点严格局部极小（也是全局极小，因为 $f$ 是凸二次型）。条件数 $3/1=3$：等高线是椭圆，长短轴比 $\sqrt3\approx1.73$。

若把函数换成 $f(x,y)=x^2+0.01y^2$，则 $H=\operatorname{diag}(2,0.02)$，条件数 $2/0.02=100$。梯度下降会在「几乎平坦」的 $y$ 方向上爬得极慢——8.7 节会看到这如何决定收敛速度。

### 5.4 场景闭环：12 万变量、115 GB 的 Hessian 怎么用

**输入**：12 万个密度变量；刚度矩阵 $K$ 稀疏（约 $10^7$ 个非零元），载荷 $\mathbf{f}$ 给定；60 次设计迭代的预算。

**计算**：不组装 $H$，只算 Hessian–向量积 $H\mathbf{v}$。对柔度问题，一次 HVP 可以写成两次线性求解（前向 + 伴随），成本约 2 倍的单次 $K\mathbf{u}=\mathbf{f}$。12 万未知数的稀疏 Cholesky 约 $10^9$ FLOP、0.1 s，所以一次 HVP 约 0.2 s。

用它跑 20 步幂迭代估计 $\lambda_{\max}$：$20\times0.2=4$ s，存储只多两个 12 万维向量（各 0.96 MB）。

**输出**：$\lambda_{\max}$ 与 $\lambda_{\min}$ 的估计（后者用移位反迭代），据此判断当前设计点离奇异有多近。若 $\lambda_{\min}/\lambda_{\max}<10^{-6}$，就切换到 MMA/OC，并加密度过滤。

**成本**：4 s/次迭代 vs 显式 Hessian 的 115 GB 内存 + 1.6 小时分解；内存从 115 GB 降到 **不到 2 MB**，差 $6\times10^4$ 倍。60 次迭代的曲率监控总共 4 分钟。

**失败条件与兜底**：

- **幂迭代只看得到 $|\lambda|_{\max}$**：$H$ 不定时，它收敛到绝对值最大的特征值，符号可能是负的。兜底：同时跑移位反迭代，或直接检查 $\mathbf{v}^\top H\mathbf{v}$ 的符号。
- **$\lambda_{\min}$ 估不准**：接近奇异时收敛极慢。兜底：用 $H+\tau I$ 的 Cholesky 是否成功作为「正定性」的布尔判断，比估具体数值更稳。
- **棋盘格**：二阶信息救不了离散化病态。兜底：密度过滤 + 周长约束。

## 6. 常见误区

1. **误区**：Hessian 一定对称。
   **纠正**：仅在二阶偏导连续时对称。数值计算中常有浮点误差，实践里常显式用 $(H+H^\top)/2$ 对称化。

2. **误区**：Hessian 正定 ⇒ 全局极小。
   **纠正**：只保证该点是严格局部极小。要全局，需要 $H$ 在整个定义域半正定（凸性，8.6 节）。

3. **误区**：特征值都为正就没问题。
   **纠正**：特征值都很小或差距极大（病态）时，优化会非常慢。条件数比「是否正定」更能预测训练难度。

4. **误区**：Hessian 和 Jacobian 是两套东西。
   **纠正**：Hessian 就是梯度的 Jacobian。标量函数的二阶导 = 一阶导（向量场）的 Jacobian。

## 7. 概念网络

- 前置：6.5 二次型 —— $\mathbf{x}^\top A\mathbf{x}$ 的二阶结构就是 $A$。
- 前置：[8.3 Jacobian](03-Jacobian.md) —— Hessian 是梯度的 Jacobian。
- 后续：[8.5 矩阵求导](05-矩阵求导与链式法则.md) —— 用矩阵微积分批量算二阶导。
- 后续：[8.6 凸性](06-凸性.md) —— $H\succeq0$ 处处成立就是凸函数。
- 后续：[8.7 牛顿法](07-梯度下降与牛顿法.md) —— 用 $H^{-1}$ 修正梯度方向。

## 8. 通往大模型：损失曲面的曲率与「锐利极小」

**出现位置**：训练动力学分析、二阶优化、锐利度（sharpness）研究。对参数 $\boldsymbol\theta\in\mathbb{R}^{P}$，损失 $L(\boldsymbol\theta)$ 的 Hessian 是 $P\times P$ 矩阵。

**形状有多夸张**：$P=7\times10^9$ 时，完整 Hessian 有 $4.9\times10^{19}$ 个元素，任何硬件都存不下。所以实践从不显式构造它。

**为什么需要**：Hessian 的最大特征值 $\lambda_{\max}$ 给出学习率上界（$\eta<2/\lambda_{\max}$，8.7 节）；特征值谱的形态区分「平坦极小」与「锐利极小」。一类常见的假设认为，泛化更好的解往往位于较平坦的区域——这类分析都建立在二阶信息上。

**可检验结论**：Hessian–向量乘积（Hessian-vector product, HVP）$H\mathbf{v}$ 可以在约两倍反向传播的成本内算出，不需要 $H$ 本身。用幂迭代估计 $\lambda_{\max}$（6.6 节的思路），就能在 7B 模型上监控曲率，而不必存下 $4.9\times10^{19}$ 个数。

**如果 Hessian 相关的步骤算错，训练时看到什么**：把不定 Hessian 直接当正定用（不做修正），更新方向可能上升，loss 在前几步突然抬升然后卡住；用幂迭代估 $\lambda_{\max}$ 却只跑了 3 步，估计值偏小、学习率上界被高估，训练出现周期性尖峰。这两种现象和「学习率太大」几乎一样，区别在于它们只在某些层、某些 batch 上出现。

## 9. 练习

1. **概念**：为什么 Hessian 的特征值能描述「曲率」？特征向量对应什么？
2. **计算**：$f(x,y)=x^2+4xy+y^2$，求 Hessian 与特征值，判断原点是极小、极大还是鞍点。
3. **计算**：$f(x,y)=2x^2+y^2$，在 $(1,1)$ 处用二阶 Taylor 估计 $f(1.1,0.9)$，并与精确值比较。
4. **场景**：最小二乘 $f(\boldsymbol\theta)=\frac12\|X\boldsymbol\theta-\mathbf{y}\|^2$，$X$ 的奇异值为 $10,1,0.1$。求 Hessian 的特征值与条件数，并说明优化难点。
5. **开放**：如果只能计算 Hessian 与任意向量的乘积，如何估计最大特征值？给出算法思路。

<details>
<summary>参考答案</summary>

1. 在驻点附近，沿单位方向 $\mathbf{u}$ 的二阶变化率是 $\mathbf{u}^\top H\mathbf{u}$；当 $\mathbf{u}$ 取特征向量时它等于对应特征值，所以特征值就是主轴曲率。特征向量是曲率主轴方向。
2. $H=\begin{bmatrix}2&4\\4&2\end{bmatrix}$，特征值 $6$ 与 $-2$，不定，原点是鞍点。
3. $f(1,1)=3$，$\nabla f=(4,2)$，$H=\operatorname{diag}(4,2)$；估计 $3+4(0.1)+2(-0.1)+\frac12(4\times0.01+2\times0.01)=3+0.4-0.2+0.03=3.23$。精确值 $2\times1.21+0.81=3.23$，一致。
4. $H=X^\top X$ 的特征值为 $100,1,0.01$；条件数 $100/0.01=10^4$。沿最小奇异值方向曲率极小，梯度下降在该方向极慢；沿最大方向学习率受 $2/100=0.02$ 限制。
5. 用幂迭代：随机 $\mathbf{v}_0$，反复 $\mathbf{v}_{k+1}=H\mathbf{v}_k/\|H\mathbf{v}_k\|$，则 $\mathbf{v}_k^\top H\mathbf{v}_k$ 收敛到 $\lambda_{\max}$（6.6 节）。每次迭代只需一次 HVP。

</details>

## 10. 一页速记

- $H_{ij}=\partial^2 f/\partial x_i\partial x_j$，二阶偏导连续时对称。
- $H=J_{\nabla f}$；二次函数 $f=\frac12\mathbf{x}^\top A\mathbf{x}$ 的 Hessian 是 $A$。
- 驻点处：正定 → 严格局部极小；负定 → 极大；不定 → 鞍点；半定 → 失效。
- 特征值 = 主曲率，条件数 = 曲面狭长程度。
- 二阶 Taylor：$f+\nabla f^\top\Delta+\frac12\Delta^\top H\Delta$。
- **物理意义 / 现实对应**：Hessian 是曲率矩阵——特征值是各主轴的「刚度」，单位是「目标单位 / 变量单位²」；工程上只用它的两个动作（HVP 与特征值估计），从不组装它。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-nocedal-wright-2006"></a>**Nocedal & Wright（2006）**：*Numerical Optimization*（2nd ed.）。Hessian、二阶充分条件与牛顿法的一阶/二阶框架，本页 §2–§4 的标准参考。[Springer](https://doi.org/10.1007/978-0-387-40065-5)
- <a id="ref-rockafellar-1970"></a>**Rockafellar（1970）**：*Convex Analysis*。凸性与二阶判据的经典处理；$H\succeq0$ 与凸性的等价关系可在其中找到严格表述。[Princeton University Press](https://doi.org/10.1515/9781400873173)
- <a id="ref-magnus-neudecker-2019"></a>**Magnus & Neudecker（2019）**：*Matrix Differential Calculus with Applications in Statistics and Econometrics*（4th ed.）。Hessian 作为「梯度的 Jacobian」的矩阵求导规则，本页 §2.3 直接使用。[Wiley](https://doi.org/10.1002/9781119541219)

---

下一篇 → [8.5 矩阵求导与链式法则](05-矩阵求导与链式法则.md)
