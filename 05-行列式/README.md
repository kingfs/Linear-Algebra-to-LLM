# M05 · 行列式

> **一句话**：行列式是线性变换对「有向体积」的缩放倍率；它用一个数回答「体积变几倍、有没有压扁、有没有翻面」。

## 这个模块解决什么

大模型里很少直接调用 `det`，但「变量替换」和「体积缩放」无处不在：概率密度换元要乘 Jacobian 行列式，归一化流的训练目标里有一项 $\log|\det J|$，多元高斯的密度里有 $\log\det\Sigma$。

本模块从几何出发，回答四个问题：

- $\det$ 到底在量什么？为什么有正负？（5.1）
- 为什么交换两行要变号？怎么算得动 $1000\times1000$ 的行列式？（5.2）
- 行列式如何判定可逆？伴随矩阵和 Cramer 法则是什么？（5.3）
- 非线性换元时，体积元怎么变？极坐标的 $r$ 从哪来？（5.4）

## 概念地图

```mermaid
flowchart LR
    A[5.1 体积的缩放因子] --> B[5.2 性质与计算]
    B --> C[5.3 逆与 Cramer]
    A --> D[5.4 Jacobian 与换元]
    C --> D
```

## 学习顺序

| 顺序 | 文档 | 时间 | 读完你应该能…… |
|---|---|---|---|
| 1 | [5.1 行列式：体积的缩放因子](01-行列式几何.md) | 30 min | 用面积/体积解释 $\det$，说出负号的几何含义 |
| 2 | [5.2 行列式的性质与计算](02-行列式的性质与计算.md) | 40 min | 用行变换和按行展开算 $3\times3$，解释交换变号 |
| 3 | [5.3 逆、Cramer 法则与伴随矩阵](03-逆与Cramer法则.md) | 30 min | 用伴随矩阵求逆，写出 Cramer 公式并说明其局限 |
| 4 | [5.4 行列式与 Jacobian](04-行列式与Jacobian.md) | 35 min | 推导极坐标的 $r$，写出密度变换与 flow 的 log-det |

## 核心公式速览

| 场景 | 公式 |
|---|---|
| $2\times2$ | $\det\begin{bmatrix}a&b\\c&d\end{bmatrix}=ad-bc$ |
| 按行展开 | $\det A=\sum_j a_{ij}(-1)^{i+j}M_{ij}$ |
| 消元 | $\det A=(-1)^s\prod_i u_{ii}$ |
| 逆 | $A^{-1}=\operatorname{adj}(A)/\det A$ |
| Cramer | $x_i=\det(A_i)/\det A$ |
| 换元 | $d\mathbf{y}=|\det J_g(\mathbf{u})|\,d\mathbf{u}$ |
| 极坐标 | $dx\,dy=r\,dr\,d\theta$ |
| 流模型 | $\log p_X(x)=\log p_Z(z)-\log|\det J_f(z)|$ |

## 本模块的大模型钩子

- **概率密度变换**：线性层 $W$ 改变体积，密度要除以 $|\det W|$（5.1、5.4）
- **归一化流**：训练目标含 $\log|\det J|$；coupling 层用三角 Jacobian 把它降到 $O(D)$（5.4）
- **多元高斯**：$\log p(x)$ 含 $-\frac12\log\det\Sigma$，用 Cholesky 算（5.3）
- **数值可逆性**：$\det$ 小不等于病态，判断要看条件数/最小奇异值（5.1、5.2）
- **高维下溢**：$0.1^{1000}$ 在 float64 中是 0，必须用 `slogdet` 或 $\sum\log|u_{ii}|$（5.2）
- **Cramer 的定位**：理论优雅、$O(n^4)$ 且不稳定，实际用 LU/QR/SVD（5.3）

## 自测（读完再做）

1. $A=\begin{bmatrix}2&1\\1&3\end{bmatrix}$ 的 $\det$ 是多少？单位正方形经 $A$ 变换后面积是多少？定向翻转了吗？
2. 说出「$\det A=0$」的至少三个等价说法，并给出几何解释。
3. 为什么交换矩阵的两行，行列式要变号？用「有向面积」解释，不要只背规则。
4. 用按行展开计算 $\det\begin{bmatrix}1&2&0\\0&1&1\\1&0&1\end{bmatrix}$。
5. 用伴随矩阵求 $B=\begin{bmatrix}3&1\\1&2\end{bmatrix}$ 的逆，并解 $B\mathbf{x}=(5,5)^\top$。
6. 极坐标变换的 Jacobian 行列式为什么是 $r$？它如何解释 $dx\,dy=r\,dr\,d\theta$？

<details>
<summary>自测答案</summary>

1. $\det=2\times3-1\times1=5$；面积变为 5 倍；$\det>0$，定向不变。
2. 列线性相关；$\operatorname{rank}(A)<n$；存在非零 $\mathbf{x}$ 使 $A\mathbf{x}=\mathbf{0}$；不可逆；$n$ 维体积被压成 0（降维）。
3. 两行是有向面积的两条边；交换它们只改变「从第一条边转到第二条边」的方向，面积大小不变、符号相反，所以 $\det$ 变号。
4. 沿第一行：$1(1\cdot1-1\cdot0)-2(0\cdot1-1\cdot1)+0=1+2=3$。
5. $\operatorname{adj}(B)=\begin{bmatrix}2&-1\\-1&3\end{bmatrix}$，$B^{-1}=\frac15\begin{bmatrix}2&-1\\-1&3\end{bmatrix}$；$\mathbf{x}=B^{-1}(5,5)^\top=(1,2)^\top$。
6. $J=\begin{bmatrix}\cos\theta&-r\sin\theta\\ \sin\theta&r\cos\theta\end{bmatrix}$，$\det J=r(\cos^2\theta+\sin^2\theta)=r$；半径 $r$ 处角宽度 $d\theta$ 对应弧长 $r\,d\theta$，所以面积元多出因子 $r$。

</details>

## 习题与实验

- [m05 习题](../exercises/m05-习题.md)：概念、计算、大模型、开放四类，含 $3\times3$ 与 Jacobian。
- [m05 答案](../exercises/m05-答案.md)：详细步骤，建议做完再对。
- [lab05 行列式](../labs/lab05_行列式.py)：用 NumPy 验证面积/体积比、$\det(AB)=\det A\det B$、奇异矩阵与条件数。

```bash
uv run --with numpy python labs/lab05_行列式.py
```

## 学完之后

- 想继续：[6.1 特征值与特征向量](../06-特征值/01-特征值与特征向量.md)，看 $\det(A-\lambda I)$ 如何定义特征多项式。
- 想补基础：回到 [1.3 点积](../01-向量与几何/03-点积.md) 复习面积推导中用到的内积。
- 想动手：把 lab05 的随机矩阵换成 $1000\times1000$，观察 `det` 与 `slogdet` 的差别。

---

下一篇 → [5.1 行列式：体积的缩放因子](01-行列式几何.md)
