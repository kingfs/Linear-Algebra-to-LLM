"""Lab11 · 一次 Transformer forward 的形状链

对应章节：11.1–11.11

验证：
1. embedding 查表 → [B,T,D]；
2. 线性层参数量与 FLOPs；
3. 单头/多头 attention 的形状、因果 mask、行和为 1；
4. RoPE 的内积只依赖相对位置；
5. LayerNorm / RMSNorm 的统计量；
6. KV Cache 显存公式；
7. LoRA 的低秩更新与零初始化；
8. 量化/反量化误差；
9. 温度对 softmax 的影响。

运行：uv run --with numpy python lab11_Transformer形状链.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------- 1
section("1. Embedding：离散 ID → [B,T,D]")
V, D, B, T = 100, 16, 2, 6
E = rng.normal(0, 0.02, size=(V, D))
ids = rng.integers(0, V, size=(B, T))
X = E[ids]
print(f"ids.shape = {ids.shape}")
print(f"X.shape   = {X.shape}  (B,T,D)")
print(f"参数量 V*D = {V * D}")


# ---------------------------------------------------------------- 2
section("2. 线性层：参数量与 FLOPs")
D_in, D_out = 16, 64
W = rng.normal(0, 1 / np.sqrt(D_in), size=(D_out, D_in))
b = np.zeros(D_out)
Y = X @ W.T + b
params = D_in * D_out + D_out
flops = 2 * B * T * D_in * D_out
print(f"X {X.shape} @ W.T {W.T.shape} -> Y {Y.shape}")
print(f"参数 = {params}，前向 FLOPs ≈ {flops}")


# ---------------------------------------------------------------- 3
section("3. 多头 attention：形状与因果 mask")
H = 4
dk = D // H
Wq = rng.normal(0, 1 / np.sqrt(D), size=(D, D))
Wk = rng.normal(0, 1 / np.sqrt(D), size=(D, D))
Wv = rng.normal(0, 1 / np.sqrt(D), size=(D, D))
Wo = rng.normal(0, 1 / np.sqrt(D), size=(D, D))

Q = (X @ Wq).reshape(B, T, H, dk).transpose(0, 2, 1, 3)  # [B,H,T,dk]
K = (X @ Wk).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
Vv = (X @ Wv).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
print(f"Q,K,V shape = {Q.shape}  (B,H,T,dk)")

scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk)  # [B,H,T,T]
mask = np.triu(np.ones((T, T), dtype=bool), k=1)     # 上三角为 True
scores_masked = np.where(mask, -1e9, scores)
A = softmax(scores_masked, axis=-1)
print(f"scores.shape = {scores.shape}")
print(f"每行和（前 3 行）= {A[0, 0, :3].sum(axis=-1)}")
print(f"因果 mask 下三角外是否全 0: {np.allclose(A[0, 0][mask], 0)}")

head_out = A @ Vv                                   # [B,H,T,dk]
concat = head_out.transpose(0, 2, 1, 3).reshape(B, T, D)
out = concat @ Wo
print(f"head_out {head_out.shape} -> concat {concat.shape} -> out {out.shape}")


# ---------------------------------------------------------------- 4
section("4. RoPE：内积只依赖相对位置")

def rope(x, positions, base=10000.0):
    """x: [T, d]；positions: [T]"""
    d = x.shape[-1]
    i = np.arange(0, d, 2)
    theta = base ** (-i / d)
    ang = positions[:, None] * theta          # [T, d/2]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


q = rng.normal(size=(1, 8))
k = rng.normal(size=(1, 8))

def inner(p, r):
    qq = rope(q, np.array([p]))
    kk = rope(k, np.array([r]))
    return (qq @ kk.T).item()

print(f"pos (5,9)   inner = {inner(5, 9):.6f}")
print(f"pos (105,109) inner = {inner(105, 109):.6f}")
print(f"pos (5,10)  inner = {inner(5, 10):.6f}（相对距离不同，应不同）")
print("结论：相对距离相同，内积相同；RoPE 编码的是相对位置。")


# ---------------------------------------------------------------- 5
section("5. LayerNorm 与 RMSNorm")
x = np.array([1.0, 2.0, 3.0])
mu, var = x.mean(), x.var()
ln = (x - mu) / np.sqrt(var + 1e-6)
rms = np.sqrt((x ** 2).mean() + 1e-6)
rn = x / rms
print(f"x = {x}")
print(f"LayerNorm = {ln}，均值 {ln.mean():.2e}，方差 {ln.var():.4f}")
print(f"RMSNorm   = {rn}，范数 {np.linalg.norm(rn):.4f}（应 ≈ sqrt(3)={np.sqrt(3):.4f}）")


# ---------------------------------------------------------------- 6
section("6. KV Cache 显存")
def kv_mib(B, Hkv, T, dk, L, bytes_per=2):
    elements = 2 * B * Hkv * T * dk
    return elements * bytes_per * L / 1024 ** 2

print(f"MHA  B=1,H=32,T=4096,dk=128,L=32 : {kv_mib(1,32,4096,128,32):.0f} MiB")
print(f"GQA  B=1,H=8, T=4096,dk=128,L=32 : {kv_mib(1,8,4096,128,32):.0f} MiB")
print(f"MHA  B=8,H=32,T=4096,dk=128,L=32 : {kv_mib(8,32,4096,128,32):.0f} MiB")


# ---------------------------------------------------------------- 7
section("7. LoRA：零初始化与参数量")
D2, r = 4096, 16
B_lora = np.zeros((D2, r))
A_lora = rng.normal(0, 1 / np.sqrt(r), size=(r, D2))
delta = B_lora @ A_lora
print(f"ΔW shape = {delta.shape}，初始 ||ΔW|| = {np.linalg.norm(delta):.1f}")
print(f"LoRA 参数 = {2 * D2 * r}；全参数 = {D2 * D2}；占比 = {2*r/D2:.4%}")


# ---------------------------------------------------------------- 8
section("8. 量化误差")
w = np.array([-0.5, -0.1, 0.0, 0.1, 0.37, 0.5])
s = 0.5 / 127
q = np.clip(np.round(w / s), -127, 127)
w_hat = q * s
print(f"w       = {w}")
print(f"q       = {q.astype(int)}")
print(f"w_hat   = {w_hat}")
print(f"绝对误差 = {np.abs(w_hat - w)}（上界 s/2 = {s/2:.5f}）")


# ---------------------------------------------------------------- 9
section("9. 温度与 softmax")
logits = np.array([2.0, 1.0, 0.0])
for temp in [0.5, 1.0, 2.0]:
    p = softmax(logits / temp)
    print(f"T={temp}: p={p}，熵={-np.sum(p*np.log(p)):.4f}")

section("10. 小结")
print("1. 形状链：[B,T] -> [B,T,D] -> [B,H,T,dk] -> [B,H,T,T] -> [B,T,D]。")
print("2. 因果 mask 在 softmax 前加 -inf，权重下三角且行和为 1。")
print("3. RoPE 使内积只依赖相对位置。")
print("4. KV Cache 与 Hkv、T、L 成正比；GQA 直接省缓存。")
print("5. LoRA 初始 ΔW=0；参数量 2Dr。")
print("6. 量化误差上界 s/2；温度控制分布尖锐度。")
