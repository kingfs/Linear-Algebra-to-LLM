---
id: 11.5
title: 位置编码与 RoPE 的旋转矩阵
module: M11
slug: rope
prereq: [1.6, 6.4]
minutes: 35
level: 核心
tags: [位置编码, RoPE, 旋转矩阵, 相对位置]
llm_hooks: [rope, long-context, position]
---

# 11.5 位置编码与 RoPE 的旋转矩阵

> **本节位置**：模块 11 · 大模型专题 ｜ **前置**：[1.6](../01-向量与几何/06-正交与投影.md) ｜ **预计**：35 分钟
>
> **一句话**：attention 本身不知道词的顺序；[RoPE](#ref-su-2021) 用「按位置旋转 Q/K」的方式把相对位置写进点积里，而且不增加任何参数。

## 0. 场景引入

「猫追狗」和「狗追猫」的 token 集合相同，attention 只看内容时输出一样——顺序信息必须由外部注入。早期 Transformer 把位置向量直接加到 embedding 上（[Vaswani 等, 2017](#ref-vaswani-2017)）：$H^{(0)}=E[T]+P$（$P$ 是位置向量），而 [RoPE](#ref-su-2021) 换成了旋转。

现在有一个具体现场：一个代码助手要给整个仓库做问答，仓库是 3.2 万 token 的源码加文档；模型训练时的上下文只有 8000，$d_k=128$，$\text{base}=10^4$，32 层。

上线前要拍板：把位置 0 到 31999 直接喂进去（裸外推），还是先把位置除以 4（位置插值 PI），还是把 base 从 1 万调到 4 万（NTK-aware 缩放）？

裸外推的代价可以算出来。最低频那一对的角速度是 $\theta_{63}=(10^4)^{-126/128}\approx1.16\times10^{-4}$ rad/token；训练最长 8000，它只转过 $0.92$ rad。推理到 32000，同一维要转到 $3.7$ rad——模型从没见过这个角度，注意力打分会系统性偏移。表现不是报错，而是**复读**：助手在 2 万 token 之后开始反复输出同一行 import。

位置插值把角度压回 $0.92$ rad，代价是相邻 token 的角差也缩小 4 倍，代码缩进这类短距离线索先变模糊。重训一个 32k 上下文的模型，按 8 张卡跑两周、每卡每小时 8 元估算，约 2.1 万元——这才是绕不开的成本。

**问题**：为什么旋转能让点积只依赖相对位置？插值到底改了什么、没改什么？

读完本节，你能回答的三个问题是：

1. RoPE 用旋转代替相加，换来了什么代数性质；
2. 相对位置性质怎么从 $R_m^\top R_n$ 推出来；
3. 长上下文外推时，插值、调 base、重训各自的代价与失效边界。

## 1. 直觉

想象每个 token 的 Q/K 是一根指针。位置 0 不旋转，位置 1 转 $\theta$，位置 2 转 $2\theta$……

当位置 $m$ 的 query 与位置 $n$ 的 key 做点积时，两者的夹角里包含 $(m-n)\theta$。**绝对位置被旋转「抵消」了，只剩相对差。**

> 💡 **一句话直觉**：**RoPE = 按位置旋转 Q/K；点积自动变成「相对位置的函数」。**

类比边界：旋转是正交变换，保长度、保夹角；它不改变向量的「内容」，只改变方向。不同维度对（频率）可以有不同的 $\theta$。

### 1.1 它在现实世界里对应什么

RoPE 像一排转速不同的表盘：每一对维度是一只指针，位置是时间，第 $i$ 只表盘的角速度是 $\theta_i$；token 在位置 $m$ 时，指针转过 $m\theta_i$。两个 token 的匹配分数只看两只指针的夹角，而夹角只由「时间差」$n-m$ 决定。

量级上，$d_k=128$、base $=10^4$ 时，最快的表盘 $\theta_0=1$ rad/token，转一圈只要 $2\pi\approx6.3$ 个 token；最慢的 $\theta_{63}=(10^4)^{-126/128}\approx1.16\times10^{-4}$，转一圈要约 $2\pi/\theta_{63}\approx5.4\times10^4$ 个 token。快表盘分辨相邻词序，慢表盘分辨段落级远近——这就是「多尺度」的具体含义。

**这个类比在哪里失效**：表盘可以无限精确地读数，而 fp32 存 $\cos(m\theta)$ 时，$m$ 大、$\theta$ 小，相位分辨率会被尾数吃掉；更重要的是，模型只在训练见过的角度区间里学过「怎么读表」，转到新角度并不自动正确。

## 2. 定义

### 2.1 二维旋转

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
$$

性质：$R(\theta)^\top R(\phi)=R(\phi-\theta)$，$\det R=1$，$R^\top=R^{-1}$。

### 2.2 RoPE 的频率

把 $d_k$ 维分成 $d_k/2$ 个二维对。第 $i$ 对的旋转角：

$$
\theta_i=\text{base}^{-2i/d_k},\qquad i=0,1,\dots,d_k/2-1
$$

位置 $m$ 的总旋转：

$$
R_m=\operatorname{diag}\big(R(m\theta_0),R(m\theta_1),\dots\big)
$$

对 query/key：

$$
\mathbf{q}_m'=R_m\mathbf{q}_m,\qquad \mathbf{k}_n'=R_n\mathbf{k}_n
$$

### 2.3 相对位置性质

$$
(\mathbf{q}_m')^\top\mathbf{k}_n'
=\mathbf{q}_m^\top R_m^\top R_n\mathbf{k}_n
=\mathbf{q}_m^\top R_{n-m}\mathbf{k}_n
$$

右边只依赖 $n-m$，不依赖绝对位置 $m,n$。

### 2.4 实现形式

对每一对 $(x_{2i},x_{2i+1})$：

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=\begin{bmatrix}\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
$$

通常只对 Q、K 应用，不对 V 应用。

## 3. 为什么这样定义

**为什么旋转而不是相加？**

- 旋转是正交变换，不改变向量长度，数值更稳定；
- 点积自然产生相对位置项，不需要额外的相对位置表；
- 没有可学习参数，不增加模型大小；
- 不同频率的对可以编码不同尺度的位置（类似傅里叶特征）。

**相对位置不是 RoPE 的首创**：[Shaw 等, 2018](#ref-shaw-2018) 已经给注意力打分加上可学习的相对位置向量；区别在于那是「加一张表」，RoPE 是「乘一个旋转」，不需要额外的位置参数。

**为什么只对 Q/K？** 位置影响的是「谁该关注谁」，即匹配分数；V 是内容，不需要旋转。若旋转 V，输出会被位置「污染」。

### 3.1 反事实：用「加绝对位置」代替旋转

假设不用旋转，而是给 Q/K 加上位置向量：$\mathbf{q}'_m=\mathbf{q}_m+\mathbf{p}_m$，$\mathbf{k}'_n=\mathbf{k}_n+\mathbf{p}_n$。展开内积：

$$
(\mathbf{q}'_m)^\top\mathbf{k}'_n
=\mathbf{q}_m^\top\mathbf{k}_n+\mathbf{q}_m^\top\mathbf{p}_n+\mathbf{p}_m^\top\mathbf{k}_n+\mathbf{p}_m^\top\mathbf{p}_n
$$

后三项分别依赖 $n$、$m$ 和 $(m,n)$，没有一项能写成 $n-m$ 的函数。取最简单的 $\mathbf{p}_t=(t,0)$、$\mathbf{q}=\mathbf{k}=(1,0)$，分数就是 $(1+m)(1+n)$。

手算两组「相对距离都是 1」的位置：$(m,n)=(1,2)$ 得 $2\times3=6$；$(m,n)=(100,101)$ 得 $101\times102=10{,}302$。同一个相对距离，分数差了 1700 多倍；而且分数随绝对位置平方增长，两个位置相差几十，logit 就差出几千，softmax 直接饱和成 one-hot；若再省掉减最大值，$e^{10302}$ 还会溢出成 `nan`。

这就是「相加」和「旋转」的分水岭：加法把位置混进内容，内积里多出 $m$、$n$ 的交叉项；旋转是正交变换，$R_m^\top R_n=R_{n-m}$ 把交叉项结构性地消掉，且 $\|R_m\mathbf{q}\|=\|\mathbf{q}\|$，分数不会随位置膨胀。

代价也要承认：要真正做到「只依赖相对位置」，Q 和 K 必须转同样的角度、用同一组频率；只转 Q 不转 K，$R_m^\top$ 单独留在内积里，性质立刻失效（见练习 4）。

## 4. 性质与推导

### 4.1 正交性

$R_m^\top R_m=I$，所以 $\|\mathbf{q}_m'\|=\|\mathbf{q}_m\|$。RoPE 不改变 Q/K 的范数，只改变方向。

### 4.2 相对位置

上面的推导：

$$
R_m^\top R_n=R(-m\theta)R(n\theta)=R((n-m)\theta)
$$

因此内积只依赖 $n-m$。

### 4.3 远程衰减

不同频率的旋转叠加，使远距离位置的内积倾向于更小（类似「相对位置越远，相关性越弱」的先验）。这不是硬约束，而是频率设计带来的偏好。

### 4.4 外推与插值

训练长度之外，$\theta_i$ 与 $m$ 的组合可能超出训练分布。常见做法：位置插值（把 $m$ 缩放）、NTK-aware 调整 base、YaRN 等。它们改的是频率/位置映射，不改变 RoPE 的代数结构。

### 4.5 计算

对每个 token、每个 pair 做一次 2×2 旋转：$O(Td_k)$，相比 attention 的 $O(T^2d_k)$ 可忽略。

## 5. 例子

### 5.1 手算：二维旋转

设 $\mathbf{q}=(1,0)$，位置 $m=1$，$\theta=90^\circ$：

$$
\mathbf{q}'=R(90^\circ)\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}0\\1\end{bmatrix}
$$

设 $\mathbf{k}=(1,0)$，位置 $n=2$，则 $\mathbf{k}'=R(180^\circ)(1,0)=(-1,0)$。

$$
(\mathbf{q}')^\top\mathbf{k}'=0\cdot(-1)+1\cdot0=0
$$

用相对位置公式：$R_{n-m}=R(90^\circ)$，$\mathbf{q}^\top R(90^\circ)\mathbf{k}=(1,0)^\top(0,1)=0$，一致。

### 5.2 手算：两个频率

$d_k=4$，$\text{base}=10000$：

$$
\theta_0=1,\qquad \theta_1=10000^{-2/4}=0.01
$$

位置 $m=10$：第一对转 $10$ rad（约 $573^\circ$），第二对转 $0.1$ rad。
高频对区分近距离，低频对区分远距离——这就是「多尺度」。

### 5.3 场景：为什么 RoPE 不增加参数

位置编码若用可学习表，长度 $T_{\max}\times D$ 个参数，且无法外推。RoPE 只有固定的三角函数，参数量为 0，长度由 $\theta_i$ 与位置计算决定。

### 5.4 场景闭环：8k 训练、32k 推理，位置怎么映射

回到第 0 节的代码助手。$L=32$ 层，$H=32$ 个头，$d_k=128$，训练长度 8000，推理 $T=32{,}000$，fp16。

**输入**：$B=1$，$T=32{,}000$，base $=10^4$，$\theta_i=(10^4)^{-2i/128}$，$i=0,\dots,63$。

**计算**：

- RoPE 本身的乘加：每 token 每对 4 次乘法加 2 次加法（约 6 FLOP），Q、K 各一遍：$2\times T\times L\times(d_k/2)\times6\approx2\times3.2\times10^4\times32\times64\times6\approx7.9\times10^8$ FLOP，约 0.8 GFLOP。
- 同一次前向的 attention：$4T^2DL\approx4\times(3.2\times10^4)^2\times4096\times32\approx5.4\times10^{14}$ FLOP，约 540 TFLOP。RoPE 只占约 $1.5\times10^{-6}$。
- 角度检查：最低频 $\theta_{63}=(10^4)^{-126/128}\approx1.16\times10^{-4}$，位置 32000 时转 $3.7$ rad，训练最多只到 $0.92$ rad；位置插值取 $m'=m/4$，$32000/4=8000$，角度正好回到 $0.92$ rad。
- KV Cache：$2\times L\times H\times d_k\times T\times2\ \text{B}=2\times32\times32\times128\times32{,}000\times2\approx1.7\times10^{10}\ \text{B}\approx16\ \text{GiB}$。

**输出决策**：选位置插值 $s=4$ 加少量继续训练，而不是裸外推；若 16 GiB KV 仍放不下，先上 GQA 或 KV 量化，再考虑缩短上下文。

**成本**：PI 把相邻 token 的角差压缩 4 倍，最快那维从 1 rad/token 变成 0.25 rad/token，「相邻两词」的分辨率下降，短距离语序（缩进、括号配对）最先受影响；继续训练 2000 步的算力，按 8 卡跑 2 天、每卡每小时 8 元估算约 3000 元。

**失败条件与兜底**：PI 只保证角度落在训练区间，不保证 attention 在新长度上仍然锐利；超过 4 倍（如 64k）时低频维又出界。兜底有两条：NTK-aware 按频率分配缩放（高频少缩、低频多缩），或改成检索式选段——先用 BM25/向量召回挑出 8k 相关片段，从根本上不让位置出界。

## 6. 常见误区

1. **误区**：RoPE 是把位置向量加到 embedding 上。
   **纠正**：RoPE 是**旋转** Q/K，不是相加；相加式位置编码是另一类方法。

2. **误区**：RoPE 对 V 也旋转。
   **纠正**：通常只旋转 Q、K；旋转 V 会改变内容表示，破坏「value 传递内容」的语义。

3. **误区**：RoPE 编码的是绝对位置。
   **纠正**：虽然旋转角由绝对位置决定，但点积只依赖相对位置；这正是它的设计目的。

4. **误区**：RoPE 可以无限外推。
   **纠正**：代数上相对位置仍成立，但频率与训练分布不匹配会导致性能下降；需要插值/调整 base 等方法。

## 7. 概念网络

- 前置：[1.6 正交与投影](../01-向量与几何/06-正交与投影.md) —— 旋转保长度。
- 前置：[6.4 谱定理](../06-特征值/04-对称矩阵与谱定理.md) —— 正交矩阵的结构。
- 后续：[11.7 KV Cache](07-KVCache.md) —— 缓存的是旋转后的 K。
- 关联：[11.3 Attention](03-Attention-上-QKV.md) —— RoPE 作用于 Q/K。

## 8. 通往大模型：形状与实现

**形状**：RoPE 不改变形状，只改变数值：

$$
Q,K:[B,H,T,d_k]\ \longrightarrow\ Q',K':[B,H,T,d_k]
$$

**实现（成对旋转）**：

```python
def rope(x, positions, base=10000.0):
    # x: [..., T, d_k]
    d = x.shape[-1]
    i = np.arange(0, d, 2)
    theta = base ** (-i / d)             # [d/2]
    ang = positions[..., None] * theta   # [..., T, d/2]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
```

**可检验结论**：对任意位置 $m,n$，$\langle R_m\mathbf{q},R_n\mathbf{k}\rangle$ 只依赖 $n-m$。可固定 $\mathbf{q},\mathbf{k}$，改变 $m,n$ 但保持差不变，验证内积不变。

**算错时用户看到什么**：只旋转 Q 不旋转 K，相对位置性质失效，长文里代词指代会错乱，用户看到「它」指到了错误的段落；把 base 从 $10^4$ 写成 $10^3$ 或 $10^6$，各频段的角速度整体错位，短文本还正常、一到几千 token 就开始复读；cos/sin 表按训练长度预计算并缓存，推理超过该长度却复用旧表，超出部分会被静默截断成同一个位置——不报错，只是所有长距离信息都消失了；用 fp16 存大位置的相位，$m\theta$ 的有效位数不够，远距离 token 几乎无法区分。

## 9. 练习

1. **概念**：RoPE 为什么不增加参数？它和可学习位置编码的本质区别是什么？
2. **计算**：$R(30^\circ)$ 作用于 $(1,0)$，结果是什么？
3. **计算**：验证 $R(\alpha)^\top R(\beta)=R(\beta-\alpha)$（用 $2\times2$ 矩阵乘法）。
4. **场景**：如果只对 Q 旋转、不对 K 旋转，相对位置性质还成立吗？
5. **开放**：长上下文外推时，为什么「调整 base」可能有效？它改变了什么？

<details>
<summary>参考答案</summary>

1. 角度由位置和固定频率算出，无学习参数；可学习编码有一张 $T_{\max}\times D$ 表，且不能自然外推。
2. $(\cos30^\circ,\sin30^\circ)\approx(0.866,0.5)$。
3. 用和角公式展开即可；几何上先转 $\beta$ 再转 $-\alpha$，净旋转 $\beta-\alpha$。
4. 不成立；$R_m^\top$ 只作用于 Q，内积会依赖 $m$ 和 $n$ 的绝对位置组合。
5. base 控制频率间隔；增大 base 让低频变化更慢，使训练长度外的位置角变化更平缓，从而缓解分布外推问题，但不改变 RoPE 的代数形式。

</details>

## 10. 一页速记

- RoPE：对 Q/K 按位置旋转，$\theta_i=\text{base}^{-2i/d_k}$。
- $R_m^\top R_n=R_{n-m}$，点积只依赖相对位置。
- 正交、保范数、零参数、只作用于 Q/K；形状不变，按二维对旋转。
- 外推需插值/调 base，代数性质不保证泛化。
- **物理意义 / 现实对应**：每对维度是一只匀速转动的表盘，位置是时间，相对位置是两只表盘的夹角。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-su-2021"></a>**Su 等（2021）**：*RoFormer: Enhanced Transformer with Rotary Position Embedding*。RoPE 的原始论文：用二维旋转矩阵按绝对位置旋转 Q/K，使内积只依赖相对位置 $n-m$，且不引入可学习参数。[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- <a id="ref-shaw-2018"></a>**Shaw 等（2018）**：*Self-Attention with Relative Position Representations*。让相对位置进入自注意力的早期方案：给 key 加上可学习的相对位置向量（等价于在注意力打分上加位置偏置）；本页的旋转是同一目标的另一种（乘法）实现。[arXiv:1803.02155](https://arxiv.org/abs/1803.02155)
- <a id="ref-vaswani-2017"></a>**Vaswani 等（2017）**：*Attention Is All You Need*。原始 Transformer 用正弦/余弦函数把绝对位置加到 embedding 上，正是第 0 节所说「相加式位置编码」的出处。[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

下一篇 → [11.6 LayerNorm 与 RMSNorm 的几何](06-LayerNorm与RMSNorm.md)
