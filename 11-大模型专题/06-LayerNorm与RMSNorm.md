---
id: 11.6
title: LayerNorm 与 RMSNorm 的几何
module: M11
slug: layernorm-rmsnorm
prereq: [1.4, 11.2]
minutes: 30
level: 核心
tags: [LayerNorm, RMSNorm, 归一化, 残差]
llm_hooks: [pre-norm, rmsnorm, training-stability]
---

# 11.6 LayerNorm 与 RMSNorm 的几何

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[1.4](../01-向量与几何/04-范数与距离.md) ｜ **预计**：30 分钟
>
> **一句话**：归一化层把每个 token 的向量「摆正」到受控的尺度上，让深层网络不被数值漂移拖垮；LayerNorm 去均值再缩放，RMSNorm 只缩放。

## 0. 场景引入

训练一个 32 层的 Transformer，如果不加归一化，会发生什么？

每层的输出尺度可能逐渐漂移：有的层激活值变成 1000，有的变成 0.001。梯度随之爆炸或消失，训练几小时后 loss 突然变成 NaN。

归一化层就是「每层之后调一次音量」：把向量拉回可用的尺度，再继续往下传。

**问题**：LayerNorm 和 RMSNorm 具体做了什么？为什么一个减均值、一个不减？它们对几何意味着什么？

## 1. 直觉

### 1.1 LayerNorm：把每个 token 的向量「标准化」

对一个 token 的 $D$ 维向量 $\mathbf{x}$：

1. 减去均值：让向量**中心化**，落在过原点的超平面上；
2. 除以标准差：让各维尺度一致；
3. 乘 $\gamma$ 加 $\beta$：给模型保留「恢复原尺度/偏移」的自由。

几何上，归一化后的向量落在半径为 $\sqrt D$ 的球面上（$\gamma=1,\beta=0$ 且忽略 $\epsilon$）。

### 1.2 RMSNorm：只调音量，不挪中心

RMSNorm 只除以均方根，不减均值。它假设「中心已经在原点附近」，或者「均值携带信息，不该删」。

> 💡 **一句话直觉**：**LayerNorm = 去均值 + 除以标准差；RMSNorm = 只除以均方根。**

类比边界：归一化不是「把信息变少」，而是把尺度这一自由度拿掉；$\gamma,\beta$ 可以把尺度/偏移学回来。

## 2. 定义

### 2.1 LayerNorm

对 $\mathbf{x}\in\mathbb{R}^{D}$：

$$
\mu=\frac1D\sum_{i=1}^D x_i,\qquad
\sigma^2=\frac1D\sum_{i=1}^D(x_i-\mu)^2
$$

$$
\operatorname{LN}(\mathbf{x})
=\boldsymbol{\gamma}\odot\frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}}+\boldsymbol{\beta}
$$

其中 $\boldsymbol{\gamma},\boldsymbol{\beta}\in\mathbb{R}^D$ 是可学习参数，$\epsilon$ 防止除零。

**逐 token、逐样本**计算，不跨 batch。

### 2.2 RMSNorm

$$
\operatorname{RMSNorm}(\mathbf{x})
=\boldsymbol{\gamma}\odot\frac{\mathbf{x}}{\sqrt{\frac1D\sum_{i=1}^D x_i^2+\epsilon}}
$$

没有 $\mu$，没有 $\boldsymbol{\beta}$（多数实现）。

### 2.3 Pre-Norm 与 Post-Norm

- **Post-Norm**（原始 Transformer）：$\mathbf{h}\leftarrow\operatorname{LN}(\mathbf{h}+\operatorname{Sublayer}(\mathbf{h}))$；
- **Pre-Norm**（现代大模型常用）：$\mathbf{h}\leftarrow\mathbf{h}+\operatorname{Sublayer}(\operatorname{LN}(\mathbf{h}))$。

Pre-Norm 让残差路径保持恒等，梯度更易直传，深层训练更稳。

## 3. 为什么这样定义

**为什么减均值？** 如果各维有一个很大的公共偏移（例如所有激活都加 100），后续线性层的输出会被这个偏移主导。中心化去掉这个「共模分量」。

**为什么除以标准差？** 让不同 token、不同层的尺度可比，避免某些维度数值过大导致 softmax/梯度饱和。

**为什么 RMSNorm 可以省掉均值？** 在 Transformer 的残差流里，很多信息编码在方向上，公共偏移可能不重要；而且省一次均值计算，速度更快。这是一个**工程取舍**，不是数学上等价。

**为什么用 $\gamma,\beta$？** 归一化会限制表达（例如强制单位尺度）。仿射参数让模型可以学回需要的尺度与偏移，恢复表达能力。

## 4. 性质与推导

### 4.1 LayerNorm 的输出统计量

忽略 $\epsilon$ 与仿射：

$$
\frac1D\sum_i \operatorname{LN}(\mathbf{x})_i=0,\qquad
\frac1D\sum_i \operatorname{LN}(\mathbf{x})_i^2=1
$$

即每个 token 的归一化向量均值为 0、方差为 1。

### 4.2 平移与缩放不变性

$$
\operatorname{LN}(a\mathbf{x}+c\mathbf{1})=\operatorname{LN}(\mathbf{x})\quad(a>0)
$$

因为减均值去掉 $c$，除以标准差去掉 $a$。RMSNorm 只对正缩放不变，对平移**不**不变。

### 4.3 RMSNorm 输出的范数

若 $\boldsymbol{\gamma}=\mathbf{1}$，则

$$
\|\operatorname{RMSNorm}(\mathbf{x})\|_2
=\frac{\|\mathbf{x}\|}{\|\mathbf{x}\|/\sqrt D}=\sqrt D
$$

所以输出落在半径 $\sqrt D$ 的球面上。

### 4.4 梯度

LayerNorm 的 Jacobian 不是对角矩阵（均值/方差依赖所有维度），反向传播需要把梯度「投影」掉均值和径向分量。RMSNorm 的 Jacobian 也类似，但少了均值项。

### 4.5 参数量与计算

- LayerNorm：$2D$ 参数；
- RMSNorm：$D$ 参数；
- 每 token 计算 $O(D)$，相对矩阵乘可忽略，但推理时是**内存带宽**操作（第 11.7 节）。

## 5. 例子

### 5.1 手算：LayerNorm

$\mathbf{x}=(1,2,3)$，$D=3$：

$$
\mu=2,\qquad \sigma^2=\frac{(1-2)^2+0+(3-2)^2}{3}=\frac23\approx0.667
$$

$$
\operatorname{LN}(\mathbf{x})=\frac{(-1,0,1)}{\sqrt{2/3}}
\approx(-1.225,0,1.225)
$$

验证均值 0，方差 $1$。

### 5.2 手算：RMSNorm

$$
\operatorname{RMS}=\sqrt{\frac{1+4+9}{3}}=\sqrt{\frac{14}{3}}\approx2.160
$$

$$
\operatorname{RMSNorm}(\mathbf{x})=\frac{(1,2,3)}{2.160}\approx(0.463,0.926,1.389)
$$

输出范数：$\sqrt{0.214+0.857+1.929}=\sqrt3\approx1.732=\sqrt D$。

### 5.3 场景：为什么大模型偏爱 Pre-Norm + RMSNorm

- Pre-Norm 让残差路径是「恒等 + 小扰动」，深层梯度不衰减；
- RMSNorm 比 LayerNorm 少一次均值、少一组参数，在数十亿次调用中节省可观；
- 代价是理论上表达略弱，但大规模训练中通常可接受。

## 6. 常见误区

1. **误区**：LayerNorm 是在 batch 维度上归一化。
   **纠正**：那是 BatchNorm。LayerNorm 对每个样本的**特征维**归一化，与 batch 大小无关。

2. **误区**：RMSNorm 和 LayerNorm 只是写法不同，效果一样。
   **纠正**：RMSNorm 不减均值，对平移敏感；两者在残差流中的行为不同。

3. **误区**：归一化会永久丢失信息。
   **纠正**：$\gamma,\beta$ 可以恢复尺度/偏移；且归一化只去掉「共模」与「整体尺度」两个自由度。

4. **误区**：$\epsilon$ 可以随便设。
   **纠正**：太小会数值不稳，太大会扭曲小范数向量；通常 $10^{-6}$ 到 $10^{-5}$。

## 7. 概念网络

- 前置：[1.4 范数](../01-向量与几何/04-范数与距离.md) —— 均方根与 L2。
- 前置：[11.2 线性层](02-线性层.md) —— 归一化后的变换。
- 后续：[11.7 KV Cache](07-KVCache.md) —— 归一化是带宽瓶颈之一。
- 后续：[11.12 训练稳定性](12-反向传播与稳定性.md) —— 归一化与梯度。
- 关联：[9.4 白化](../09-概率与统计/04-白化.md) —— 去相关 vs 去尺度。

## 8. 通往大模型：形状与实现

**形状不变**：`[B,T,D] → [B,T,D]`，逐 token 计算。

```python
def rms_norm(x, gamma, eps=1e-6):
    # x: [..., D]
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return gamma * x / rms
```

**为什么它是推理瓶颈**：每个 token 都要读一遍 $D$ 维向量、算平方和、再写回；在 decode 阶段，矩阵乘是「瘦」的，归一化/残差等逐元素操作的**内存带宽**占比反而很高。

**可检验结论**：RMSNorm 输出（$\gamma=1$）的 L2 范数为 $\sqrt D$，与输入尺度无关。若你的实现输出范数随输入增大，说明归一化写错了。

## 9. 练习

1. **概念**：LayerNorm 和 BatchNorm 的区别是什么？为什么 NLP 常用 LayerNorm？
2. **计算**：$\mathbf{x}=(2,4,6)$，求 LayerNorm（忽略 $\epsilon,\gamma,\beta$）和 RMSNorm 的输出。
3. **计算**：$D=4096$，LayerNorm 与 RMSNorm 各有多少参数？
4. **场景**：Pre-Norm 为什么比 Post-Norm 更容易训练深层网络？从残差路径解释。
5. **开放**：如果去掉所有归一化层，但把初始化尺度调得很小，能否替代归一化？可能有什么问题？

<details>
<summary>参考答案</summary>

1. BatchNorm 跨 batch 统计，依赖 batch 大小、训练/推理不一致；LayerNorm 对每个样本独立，适合变长序列。
2. $\mu=4$，$\sigma^2=8/3$，LN$=(-1.225,0,1.225)$；RMS$=\sqrt{56/3}\approx4.320$，RMSNorm$\approx(0.463,0.926,1.389)$。
3. LayerNorm $2D=8192$；RMSNorm $D=4096$。
4. Pre-Norm 中残差主路径是恒等映射，梯度可以直接回传；Post-Norm 每层都经过归一化，深层连乘更易衰减/爆炸。
5. 部分可以（如 Fixup 初始化），但训练动态对超参更敏感，且不同层尺度仍可能漂移；归一化提供了更鲁棒的尺度控制。

</details>

## 10. 一页速记

- LayerNorm：减均值、除标准差、仿射；输出均值 0 方差 1。
- RMSNorm：只除均方根；$\gamma=1$ 时输出范数 $\sqrt D$。
- 都是逐 token、逐样本，不跨 batch。
- Pre-Norm 保残差恒等路径，深层更稳。
- 归一化是低 FLOPs、高带宽的操作。

---

下一篇 → [11.7 KV Cache：形状、显存与带宽](07-KVCache.md)
