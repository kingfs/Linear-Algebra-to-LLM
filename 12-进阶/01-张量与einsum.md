---
id: 12.1
title: 张量与 einsum：把形状写成公式
module: M12
slug: einsum
prereq: [0.3, 11.3]
minutes: 40
level: 核心
tags: [张量, einsum, 广播, attention]
llm_hooks: [attention-einsum, shape-debug]
---

# 12.1 张量与 einsum：把形状写成公式

> **本节位置**：模块 12 · 进阶 ｜ **前置**：[0.3](../00-准备/03-标量向量矩阵张量.md)、[11.3](../11-大模型专题/03-Attention-上-QKV.md) ｜ **预计**：40 分钟
>
> **一句话**：einsum 用「给每个轴起名字」的方式写张量运算；只要名字对得上，形状就不会错，代码也能直接读出数学。

## 0. 场景引入

你想写多头 attention，形状是：

- $Q$：`[B, H, T, d_k]`
- $K$：`[B, H, T, d_k]`
- 想要 $S$：`[B, H, T, T]`，其中 $S_{bhij}=\sum_d Q_{bhid}K_{bhjd}$

用 `reshape` + `transpose` + `matmul` 写，很容易在某一步把 $T$ 和 $d_k$ 弄反，直到训练报错才发现。

用 einsum：

```python
S = np.einsum("bhid,bhjd->bhij", Q, K)
```

**一句话读出来**：对 $d$ 求和，保留 $b,h,i,j$。形状一目了然。

**问题**：einsum 的规则是什么？为什么它能统一矩阵乘、转置、求和、外积？有没有性能陷阱？

## 1. 直觉

einsum 的核心是「轴的名字」：

- 每个输入张量的每个轴起一个字母；
- 输出里出现的字母 = 要保留的轴；
- 输入里出现、输出里没出现的字母 = 要求和的轴；
- 同一个字母在不同输入里出现 = 这些轴必须对齐（同一种索引）。

> 💡 **一句话直觉**：**einsum = 「保留哪些轴、求和哪些轴」的声明式写法；字母就是轴的身份证。**

类比边界：einsum 不是万能的——它不表达非线性、广播规则也不如 `+`/`*` 灵活；复杂算子仍要手写。

## 2. 定义

### 2.1 张量

$n$ 阶张量是带 $n$ 个索引的数组：

$$
X_{i_1 i_2\cdots i_n}\in\mathbb{R}^{d_1\times\cdots\times d_n}
$$

标量 0 阶，向量 1 阶，矩阵 2 阶。深度学习里常见 4 阶：`[B,H,T,d]`。

### 2.2 einsum 语法

$$
\texttt{"input1, input2, ... -> output"}(A, B, \dots)
$$

规则：

1. 输入中重复的字母（同一张量内）表示对角线；
2. 跨输入重复的字母表示对应轴相乘；
3. 输出中未出现的字母求和；
4. 输出中字母顺序可任意，决定结果轴的顺序；
5. 省略 `-> output` 时，einsum 自动求和所有只出现一次的轴（不推荐，容易出错）。

### 2.3 常见运算

| 运算 | einsum |
|---|---|
| 矩阵乘 | `"ik,kj->ij"` |
| 批量矩阵乘 | `"bik,bkj->bij"` |
| 转置 | `"ij->ji"` |
| 迹 | `"ii->"` |
| 外积 | `"i,j->ij"` |
| 逐元素乘求和（内积） | `"i,i->"` |
| 行和 | `"ij->i"` |
| 注意力分数 | `"bhid,bhjd->bhij"` |
| 注意力加权 | `"bhij,bhjd->bhid"` |

## 3. 为什么这样定义

普通矩阵乘要求「左矩阵的列 = 右矩阵的行」，这个约束在批量、多头、多维时被 reshape 打乱。einsum 把「哪个轴对哪个轴」显式写出来，消除了 reshape 的顺序歧义。

它也把「求和」显式化：`"ij->i"` 明确是对 $j$ 求和。读代码时，数学公式与实现一一对应，减少「形状对了但语义错了」的 bug。

## 4. 性质与推导

### 4.1 自由索引与求和索引

- **自由索引**（free）：出现在输出中，保留；
- **求和索引**（summed/contracted）：只出现在输入中，求和。

输出形状 = 自由索引对应维度的笛卡尔积。

### 4.2 同一张量内重复索引 = 对角线

`"ii->i"` 取出对角线；`"ii->"` 取迹。这与「跨输入重复」不同，后者是乘法对齐。

### 4.3 结合顺序影响性能

einsum 可能有多种 contraction order。`"ij,jk,kl->il"` 可以先算 $ij,jk$ 再乘 $kl$，也可以先算 $jk,kl$。不同顺序的 FLOPs 可能差很多；NumPy/PyTorch 会尝试优化，但复杂表达式建议手动分步或用 `opt_einsum`。

### 4.4 与广播的区别

广播是「形状不同的数组按规则扩展」，einsum 是「索引对齐与求和」。二者可以组合，但不要混为一谈。

## 5. 例子

### 5.1 手算：矩阵乘

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
B=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
$$

`"ik,kj->ij"`：

$$
(AB)_{11}=1\cdot1+2\cdot0+3\cdot1=4,\quad
(AB)_{12}=1\cdot0+2\cdot1+3\cdot1=5
$$

$$
AB=\begin{bmatrix}4&5\\10&11\end{bmatrix}
$$

### 5.2 手算：注意力分数

$T=3,d_k=2$，$Q=K=I_3$ 的前两列（仅示意）：

$$
Q=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}
$$

`"bhid,bhjd->bhij"` 在 $B=H=1$ 时就是 $QK^\top$：

$$
QK^\top=\begin{bmatrix}1&0&0\\0&1&0\\0&0&0\end{bmatrix}
$$

第 3 个 token 与谁都正交，分数全 0。

### 5.3 场景：多头 attention 的完整 einsum

```python
# Q,K,V: [B,H,T,d]
S = np.einsum("bhid,bhjd->bhij", Q, K) / np.sqrt(d)
A = softmax(S, axis=-1)
O = np.einsum("bhij,bhjd->bhid", A, V)       # [B,H,T,d]
Y = np.einsum("bhtd,de->bhte", O.transpose(0,2,1,3).reshape(B,T,D), Wo)
```

**注意**：`"bhij,bhjd->bhid"` 中 $j$ 是求和索引，正是「对所有 key 加权」。

## 6. 常见误区

1. **误区**：einsum 一定比 matmul 快。
   **纠正**：einsum 可能产生临时数组、走通用路径；简单矩阵乘用 `@` 往往更快（底层 GEMM）。

2. **误区**：输出字母顺序无所谓。
   **纠正**：顺序决定结果轴的排列；`"ij->ji"` 与 `"ij->ij"` 不同。

3. **误区**：同一张量内重复字母和跨张量重复是一回事。
   **纠正**：前者是对角线，后者是对齐相乘。

4. **误区**：省略 `->` 更简洁。
   **纠正**：隐式模式会求和所有只出现一次的轴，容易产生意外结果；生产代码建议显式写出输出。

## 7. 概念网络

- 前置：[0.3 张量轴](../00-准备/03-标量向量矩阵张量.md) —— 形状语言。
- 前置：[11.3 Attention](../11-大模型专题/03-Attention-上-QKV.md) —— QKᵀ 与加权。
- 关联：[10.3 GEMM](../10-数值线性代数/03-BLAS与GEMM.md) —— 底层矩阵乘。
- 关联：[11.4 多头](../11-大模型专题/04-Attention-下-softmax与多头.md) —— 头维度的拆分。

## 8. 通往大模型：形状调试

**用 einsum 读 attention**：

$$
S_{bhij}=\sum_d Q_{bhid}K_{bhjd},\qquad
O_{bhid}=\sum_j A_{bhij}V_{bhjd}
$$

这两个式子直接对应两行 einsum；比 `transpose().reshape().matmul()` 更难写错。

**性能提示**：

- 简单的 `"bik,bkj->bij"` 用 `torch.bmm`/`@` 更快；
- 大 einsum 用 `opt_einsum` 找最优收缩顺序；
- 注意中间张量 $[B,H,T,T]$ 的显存（第 11.4 节）。

**可检验结论**：`"bhid,bhjd->bhij"` 的输出形状是 `[B,H,T,T]`；若你把 `d` 写进输出（`->bhijd`），会得到一个 5 阶张量，通常是 bug。

## 9. 练习

1. **概念**：einsum 中「输出里没出现的字母」意味着什么？
2. **写出 einsum**：批量矩阵乘、转置、迹、外积、行和。
3. **计算**：用 `"ij,jk->ik"` 手算 $\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix}$。
4. **场景**：写出多头 attention 中 $AV$ 的 einsum，并说明哪个字母被求和。
5. **开放**：为什么 `"ij,jk,kl->il"` 的收缩顺序会影响性能？举一个数量级例子。

<details>
<summary>参考答案</summary>

1. 该轴被求和（contracted）；输出中不会保留它。
2. `"bik,bkj->bij"`；`"ij->ji"`；`"ii->"`；`"i,j->ij"`；`"ij->i"`。
3. $\begin{bmatrix}2&1\\4&3\end{bmatrix}$。
4. `"bhij,bhjd->bhid"`；$j$（key 位置）被求和。
5. 不同顺序的中间张量大小不同；例如 $i=j=k=l=1000$ 时，先算 $ij,jk$ 得 $1000^3$ 次运算，再乘 $kl$ 又 $1000^3$；若先算 $jk,kl$ 同理，但若维度差异大，中间结果可能爆炸或缩减，FLOPs 差几个数量级。

</details>

## 10. 一页速记

- einsum：字母命名轴；输出保留自由轴，其余求和。
- 同张量重复 = 对角线；跨张量重复 = 对齐相乘。
- 常见：`ik,kj->ij`、`bhid,bhjd->bhij`、`bhij,bhjd->bhid`。
- 显式写输出，避免隐式求和陷阱。
- 简单矩阵乘用 `@`，复杂表达式再考虑 einsum/opt_einsum。

---

下一篇 → [12.2 随机矩阵理论](02-随机矩阵理论.md)
