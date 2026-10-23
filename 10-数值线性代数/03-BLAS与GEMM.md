---
id: 10.3
title: BLAS 与 GEMM：GPU 为什么快
module: M10
slug: blas-gemm
prereq: [3.3, 10.2]
minutes: 40
level: 核心
tags: [BLAS, GEMM, 算术强度, 分块, Roofline, cuBLAS]
llm_hooks: [gemm, prefill-decode, tensor-core, kv-cache-bandwidth]
---
# 10.3 BLAS 与 GEMM：GPU 为什么快
> **本节位置**：模块 10 · 数值线性代数 ｜ **前置**：[3.3 矩阵乘法](../03-矩阵与线性映射/03-矩阵乘法.md)、[10.2 矩阵范数与条件数](02-矩阵范数与条件数.md) ｜ **预计**：40 分钟
>
> **一句话**：GPU 不是「所有运算都快」，而是「每个字节被复用很多次时特别快」；GEMM 正好是这种运算的极端代表。
## 0. 场景引入
两个 $4096\times4096$ 的 fp32 矩阵相乘，要做

$$
2MNK=2\times4096^3\approx1.37\times10^{11}
$$

次浮点运算。一块峰值 $20$ TFLOPS 的 GPU 理论上 $6.9$ ms 算完；但同一块 GPU 的显存带宽只有约 $2$ TB/s，把三个矩阵各读一遍（$201$ MB）就要 $0.1$ ms。

**问题**：为什么实际耗时主要由「算」而不是「搬」决定？如果 batch 变成 1（每次只乘一个向量），结论为什么反过来？分块（tiling）到底在省什么？
## 1. 直觉
### 1.1 厨房类比：切菜 vs 取菜
把计算想成厨房：**算力**是厨师切菜的速度，**带宽**是跑腿从仓库取食材的速度。

- 如果每切一刀都要跑一趟仓库，厨师再快也没用——这是**访存受限（memory-bound）**。
- 如果一次取回一大筐菜，切很多刀才需要再取——这是**计算受限（compute-bound）**。

**算术强度（arithmetic intensity）** 就是「每从仓库取 1 字节，能切多少刀」：FLOPs / Bytes。

> 💡 **一句话直觉**：**算术强度高的运算，GPU 才快；GEMM 的强度随分块尺寸增长。**

类比边界：厨房类比忽略了并行度与延迟。GPU 有上万个线程，延迟靠切换隐藏，不只是「带宽够不够」。
### 1.2 [Roofline](#ref-williams-2009)：两个天花板取较小者
一个 kernel 的时间至少是

$$
T=\max\Big(\frac{\text{FLOPs}}{\text{峰值算力}},\ \frac{\text{Bytes}}{\text{带宽}}\Big)
$$

把算术强度画在横轴、性能画在纵轴，就是**Roofline 模型**：低强度段是斜线（带宽决定），高强度段是平台（算力决定）。两条线的交点叫**拐点（ridge point）**，其横坐标是「峰值算力 / 带宽」。
## 2. 定义
### 2.1 GEMM 的形状
**GEMM（general matrix multiply）** 是 BLAS 的 Level-3 原语：

$$
C\leftarrow \alpha AB+\beta C,\qquad
A\in\mathbb{R}^{M\times K},\ B\in\mathbb{R}^{K\times N},\ C\in\mathbb{R}^{M\times N}
$$

其中 $\alpha,\beta$ 是标量。形状记成 `[M, K] @ [K, N] -> [M, N]`；一次乘加算 2 个 FLOP（1 次乘、1 次加），所以总运算量是 $2MNK$。
### 2.2 BLAS 的三个层级
[BLAS（Basic Linear Algebra Subprograms）](#ref-lawson-1979) 是一组标准接口，按运算量分三层：

| 层级 | 典型函数 | 运算 | 复杂度 | 算术强度 |
|---|---|---|---|---|
| Level 1 | `axpy`、`dot`、`nrm2` | 向量-向量 | $O(n)$ | 很低（约 1/4–1） |
| Level 2 | `gemv`、`trsv` | 矩阵-向量 | $O(n^2)$ | 低（约 1） |
| Level 3 | `gemm`、`trsm`、`syrk` | 矩阵-矩阵 | $O(n^3)$ | 高（可随分块增长） |

Level 3 的运算量按 $n^3$ 涨、数据量按 $n^2$ 涨，所以 $n$ 越大，每个字节被复用得越多。
### 2.3 算术强度
对 GEMM，若只统计必须从显存读入的数据（$A,B,C$ 各一遍）：

$$
\text{AI}=\frac{2MNK}{4(MK+KN+MN)}\ \text{FLOP/byte}
$$

$M=N=K=n$ 时 $\text{AI}=n/6$。$n=4096$ 时是 $683$ FLOP/byte。这个数远大于拐点，所以大 GEMM 是计算受限。
### 2.4 内存层次
GPU 的存储不是一块，而是一套层次：

| 层次 | 典型容量 | 带宽量级 | 谁在用 |
|---|---|---|---|
| 寄存器（register） | 每 SM 数百 KB | 极高 | 每个线程的累加器 |
| 共享内存/L1 | 每 SM 约 100–200 KB | 高 | 一个 block 共享的 tile |
| L2 | 数十 MB | 中 | 多个 block 复用 |
| HBM（显存） | 数十 GB | 1.5–3 TB/s | 全局读写 |

分块（tiling/blocking）就是把大矩阵切成能塞进上一层的 tile，让数据在寄存器/共享内存里被多次使用。
## 3. 为什么这样定义
**为什么 BLAS 要分层？** 因为不同层级的瓶颈不同。Level 1/2 的每个浮点运算只伴随少量甚至一次数据搬运，优化空间在「少读一次」；Level 3 有 $O(n)$ 的复用潜力，优化空间在「让复用发生」。把接口标准化，硬件厂商就能为每一层写专用 kernel。

**为什么 GEMM 是深度学习的核心原语？** 线性层、卷积（im2col 后）、注意力的 $QK^\top$ 与 $PV$ 都能写成 GEMM。只要 GEMM 快，模型的绝大部分 FLOPs 就快。

**为什么列主序会出现？** BLAS 源自 Fortran，按列连续存储；NumPy 默认行主序，调用 cuBLAS 时会用转置标志适配。这解释了为什么同一个矩阵在不同库里的「连续维度」可能不同。

**为什么不用一个「万能 kernel」？** 形状差异太大：$M=1$ 的 GEMV 与 $M=N=K=4096$ 的 GEMM 瓶颈完全不同，前者访存受限、后者计算受限，必须用不同策略。
## 4. 性质与推导
### 4.1 FLOPs 与访存量
GEMM 有 $MN$ 个输出元素，每个元素要做 $K$ 次乘加，所以

$$
\text{FLOPs}=2MNK
$$

按 $C=\alpha AB$（$\beta=0$）计，$A,B$ 各读一次、$C$ 写一次，访存量是 $4(MK+KN+MN)$ 字节（fp32）。$M=N=K=4096$ 时：

$$
\text{FLOPs}=1.374\times10^{11},\qquad \text{Bytes}=4\times3\times4096^2=2.013\times10^{8}
$$
### 4.2 不分块时为什么慢
最朴素的三重循环，每做一次乘加都要取 $a_{ik}$ 与 $b_{kj}$ 各 4 字节，即 8 字节换 2 FLOP，算术强度只有 $0.25$。若完全不靠缓存，$1.37\times10^{11}$ FLOP 要搬 $5.5\times10^{11}$ 字节——按 $2$ TB/s 需要 $0.275$ s。

真实硬件有缓存，不会这么惨；但这个极端说明：**不复用就必然访存受限**。
### 4.3 分块把强度提高 $b/4$
把输出切成 $b\times b$ 的 tile。每个 tile 需要 $A$ 的一条 $b\times K$ 面板和 $B$ 的一条 $K\times b$ 面板，共 $2bK$ 个元素；tile 的个数是 $MN/b^2$。因此总访存量是

$$
4\times\frac{2bKMN}{b^2}=\frac{8MNK}{b}\ \text{字节}
$$

与 §4.1 的 FLOPs 相除，得

$$
\text{AI}=\frac{2MNK}{8MNK/b}=\frac{b}{4}
$$

读作：**tile 越大，每个字节被复用得越多**。$b=128$ 时 AI 为 $32$；$b=256$ 时为 $64$。当然，tile 不能无限大——它必须塞进共享内存/寄存器，且要留够并行度。
### 4.4 Roofline 的判据
以 A100 量级为例：fp32 峰值约 $20$ TFLOPS、显存带宽约 $2$ TB/s，拐点约 $10$ FLOP/byte。$n=4096$ 的 GEMM AI 约 $683$，远在拐点右侧，计算受限；$M=1$ 的 GEMV AI 约 $1$（bf16 权重、忽略输出读写），远在左侧，访存受限。
## 5. 例子
### 5.1 手算：$M=N=K=4096$ 的 GEMM
$$
\text{FLOPs}=2\times4096^3=1.374\times10^{11}
$$

按 $300$ TFLOPS（bf16 张量核量级）算：$1.374\times10^{11}/3\times10^{14}=0.458$ ms。按 $2$ TB/s 搬 $201$ MB：$2.013\times10^{8}/2\times10^{12}=0.101$ ms。**计算时间是访存时间的 4.5 倍**，所以优化重点是让张量核别闲着。

若不分块、按 $0.25$ FLOP/byte 估算，访存时间会变成约 $0.275$ s；分块 $b=128$ 后访存量降到 $4.3$ GB，时间约 $2.15$ ms——仍然比「理想一遍」的 $0.101$ ms 大，但已经从「百毫秒级」回到「毫秒级」。
### 5.2 手算：batch=1 时为什么反过来
解码时一次只处理一个 token：$M=1$、$N=K=4096$。此时

$$
\text{FLOPs}=2\times1\times4096\times4096=3.355\times10^{7}
$$

权重 $W\in\mathbb{R}^{4096\times4096}$ 用 bf16 存是 $4096^2\times2=33.55$ MB。忽略 $\mathbf{x}$ 与输出 $\mathbf{y}$ 的读写，算术强度约

$$
\frac{3.355\times10^{7}}{3.355\times10^{7}}\approx1\ \text{FLOP/byte}
$$

按 $2$ TB/s 读权重需要 $16.8$ µs；按 $300$ TFLOPS 算只要 $0.11$ µs。**访存是瓶颈，差了两个数量级**。把 batch 提到 $64$，FLOPs 变 $64$ 倍、权重仍读一遍，AI 升到约 $64$，依然在拐点左侧；直到 batch 足够大，才会重新变成计算受限。
### 5.3 场景：cuBLAS 在做什么
`cublasSgemm` 不会只跑一个 kernel。它根据 $M,N,K$、数据类型、转置标志选择：

- **tile 尺寸与线程块形状**：如 $128\times128\times32$，匹配共享内存与寄存器预算；
- **split-K**：$K$ 很大而 $M,N$ 很小时，把 $K$ 切给多个 block 并行，再原子累加；
- **stream-K**：把 tile 工作重新分配到固定数量的 CTA，减少尾效应；
- **张量核指令**：bf16/fp16 输入、fp32 累加，一条 MMA 指令完成一个小矩阵块。

CUTLASS 把这些策略写成模板；cuBLAS 则是闭源的高度调优版本。FlashAttention 的思路类似：把 $T\times T$ 的注意力矩阵切块，在片上完成 softmax，避免把中间矩阵写回显存。
## 6. 常见误区
1. **误区**：FLOPs 越高的运算越慢。
   **纠正**：GEMM 的 FLOPs 是 $O(n^3)$，但每个字节复用 $O(n)$ 次，实际吞吐可以接近峰值；GEMV 的 FLOPs 少，却被带宽卡死。
2. **误区**：GPU 快是因为核心多。
   **纠正**：核心多只提供峰值算力；能否达到峰值取决于算术强度、分块、寄存器占用与内存合并访问。
3. **误区**：分块越大越好。
   **纠正**：tile 受共享内存、寄存器数量、occupancy 限制；过大 tile 会让并行度不足、边界浪费增加。
4. **误区**：batch 越大，推理一定越快（每秒 token 数）。
   **纠正**：增大 batch 提高 AI，但也会增加每步延迟与显存占用；吞吐与延迟是两条曲线。
## 7. 概念网络
- 前置：[3.3 矩阵乘法](../03-矩阵与线性映射/03-矩阵乘法.md) —— GEMM 就是批量矩阵乘法。
- 前置：[10.2 矩阵范数与条件数](02-矩阵范数与条件数.md) —— 数值稳定性和精度决定能不能用低精度张量核。
- 后续：[10.5 稀疏矩阵](05-稀疏矩阵.md) —— 当大部分乘数是 0 时，GEMM 的「高复用」假设失效。
- 关联：[11.2 线性层](../11-大模型专题/02-线性层.md) —— 每个线性层都是一次 GEMM。
- 关联：[11.7 KV Cache](../11-大模型专题/07-KVCache.md) —— 解码的瓶颈从算力转向带宽。
- 关联：[11.4 Attention](../11-大模型专题/04-Attention-下-softmax与多头.md) —— $QK^\top$ 与 $PV$ 是 batched GEMM。
## 8. 通往大模型：prefill 与 decode 是两种负载
**出现位置一：线性层。** 一个 token 的隐藏状态是 $\mathbf{x}\in\mathbb{R}^{D}$，一层 FFN 是 $W_1\in\mathbb{R}^{4D\times D}$、$W_2\in\mathbb{R}^{D\times 4D}$。把 $BT$ 个 token 堆成 $X\in\mathbb{R}^{BT\times D}$，就是两次 GEMM：$XW_1^\top$ 与 $(\cdot)W_2^\top$。$D=4096$、$BT=16384$ 时，单次 GEMM 的 FLOPs 是 $2\times16384\times4096\times16384\approx2.2\times10^{12}$。

**出现位置二：prefill 是计算受限。** $B=8$、$T=2048$、$D=4096$ 时 $BT=16384$，矩阵足够「胖」，AI 远高于拐点，GPU 能跑满张量核。这时提升吞吐靠更大的 batch、更快的卡、更低的精度（bf16/fp8）。

**出现位置三：decode 是访存受限。** 每生成一个 token，$M=1$，要读完整层权重。7B 模型 bf16 约 $14$ GB，按 $2$ TB/s 读一遍是 $7$ ms，理论上限约 $143$ token/s；按 $1$ TB/s 则约 $71$ token/s。**这就是「decode 速度 ≈ 带宽 / 模型字节数」的由来。** KV Cache 也要读：$L=32$、$H=32$、$T=4096$、$d_k=128$ 时每 token 的 KV 约 $512$ KB，长上下文下它甚至超过权重流量。

**出现位置四：张量核与精度。** bf16/fp16 的 MMA 指令以 fp32 累加，吞吐通常是 fp32 CUDA 核心的数倍到十几倍。代价是输入先被舍入（第 10.1 节）；若某层 $\kappa$ 很大，低精度的误差会被放大（第 10.2 节）。这就是混合精度要保留 fp32 主权重的原因。

**可检验结论**：对固定模型，decode 的 token/s 上界近似为「显存带宽 ÷ 每步需读字节数」，与峰值 FLOPs 基本无关；prefill 的吞吐则随 $BT$ 增大而上升，直到计算受限。你可以在 lab10 里用不同 $M$ 的 GEMM 计时看到这条曲线：$M$ 从 1 增到 1024，GFLOPS 从「带宽限制」爬升到「算力平台」。
## 9. 练习
1. **概念**：为什么 BLAS Level 3 比 Level 2 更容易跑满峰值？用算术强度解释。
2. **计算**：$M=N=K=1024$ 的 fp32 GEMM，FLOPs、最小访存字节数、算术强度各是多少？在 $20$ TFLOPS/$2$ TB/s 的机器上，理论时间是算力还是带宽决定？
3. **计算**：解码时 $M=1$、$N=K=4096$，权重 bf16。算术强度约多少？若带宽 $2$ TB/s，读一遍权重最少多少时间？
4. **场景**：$B=8$、$T=2048$、$D=4096$ 的一层 FFN（$D\to4D\to D$），两次 GEMM 共多少 FLOPs？若 GPU 峰值 $300$ TFLOPS，理论最短时间是多少？
5. **开放**：分块尺寸 $b$ 越大，AI 越高，但为什么不能一直增大？列出至少三个限制，并说明它们如何影响实际 kernel。

<details>
<summary>参考答案</summary>

1. Level 2 每读一个矩阵元素只做 $O(1)$ 次乘加，AI 约 1；Level 3 的每个元素可被 $O(n)$ 个输出复用，分块后 AI 为 $b/4$。AI 越高，越容易越过 Roofline 拐点、进入算力受限区。
2. FLOPs $=2\times1024^3=2.147\times10^{9}$；字节 $=4\times3\times1024^2=1.258\times10^{7}$；AI $=2.147\times10^9/1.258\times10^7=170.7$ FLOP/byte。拐点约 $10$，所以计算受限。算力时间 $2.147\times10^9/2\times10^{13}=0.107$ ms，带宽时间 $1.258\times10^7/2\times10^{12}=6.3$ µs。
3. FLOPs $=2\times4096^2=3.355\times10^7$；权重字节 $=4096^2\times2=3.355\times10^7$；AI 约 $1$。带宽时间 $3.355\times10^7/2\times10^{12}=16.8$ µs。
4. 第一次 $2\times16384\times4096\times16384=2.2\times10^{12}$，第二次形状对称、FLOPs 相同，共约 $4.4\times10^{12}$。按 $300$ TFLOPS 理论 $14.6$ ms（实际还受访存、softmax、归一化等影响）。
5. 共享内存/寄存器容量限制 tile 大小；occupancy 与并行度会随 tile 增大而下降；边界 tile 在 $M,N$ 不能被 $b$ 整除时浪费算力；此外还有 L2 容量、银行冲突、指令级并行等。实际 kernel 要在这些约束下搜索 tile 与流水线深度。

</details>

## 10. 一页速记
- GEMM：$C=\alpha AB+\beta C$，$A:[M,K]$、$B:[K,N]$、$C:[M,N]$，FLOPs $=2MNK$。
- BLAS L1/L2/L3 分别对应向量、矩阵-向量、矩阵-矩阵；只有 L3 有高复用。
- 算术强度 = FLOPs / Bytes；Roofline 时间取「算力」与「带宽」两个上界的较大者。
- 分块让 AI 从 $0.25$ 升到 $b/4$；$b=128$ 时约 $32$ FLOP/byte。
- prefill 计算受限、decode 访存受限；decode 速度上界约为带宽除以每步读取字节数。

## 11. 参考文献与致谢

这一页的知识不是凭空来的。下面列出本页直接借用的原始工作与权威整理；正文中的人名、定理名已链接到对应条目。链接优先指向 DOI、arXiv 或可长期访问的原文存档。

- <a id="ref-lawson-1979"></a>**Lawson 等（1979）**：*Basic Linear Algebra Subprograms for Fortran Usage*。BLAS 的原始接口规范，本页 §2.2 的 Level 1/2/3 分层由此而来。[ACM Transactions on Mathematical Software](https://doi.org/10.1145/355841.355847)
- <a id="ref-dongarra-1990"></a>**Dongarra 等（1990）**：*A Set of Level 3 Basic Linear Algebra Subprograms*。Level-3 BLAS（GEMM 一族）的标准化，让厂商可以针对矩阵-矩阵乘写专用 kernel。[ACM Transactions on Mathematical Software](https://doi.org/10.1145/77626.79170)
- <a id="ref-williams-2009"></a>**Williams, Waterman & Patterson（2009）**：*Roofline: An Insightful Visual Performance Model for Multicore Architectures*。Roofline 模型与「拐点」的出处，本页 §1.2 直接使用。[Communications of the ACM](https://doi.org/10.1145/1498765.1498785)
- <a id="ref-strassen-1969"></a>**Strassen（1969）**：*Gaussian Elimination Is Not Optimal*。矩阵乘法可以少于 $O(n^3)$ 次标量乘，说明 GEMM 的「计算量」本身也是可优化的对象。[Numerische Mathematik](https://doi.org/10.1007/BF02165411)

---

下一篇 → [10.4 迭代法与共轭梯度](04-迭代法与共轭梯度.md)
