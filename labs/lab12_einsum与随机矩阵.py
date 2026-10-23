"""Lab12 · einsum、随机矩阵与全景

对应章节：12.1 einsum / 12.2 随机矩阵

验证：
1. einsum 与 matmul/transpose/trace 的等价性；
2. 批量矩阵乘与多头 attention 的 einsum 写法；
3. 随机矩阵的谱范数随缩放的变化；
4. Marchenko–Pastur 区间（模拟）。

运行：uv run --with numpy python lab12_einsum与随机矩阵.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(1)


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------- 1
section("1. einsum 基本运算")
A = rng.normal(size=(2, 3))
B = rng.normal(size=(3, 4))

C1 = np.einsum("ik,kj->ij", A, B)
C2 = A @ B
print(f"A {A.shape} @ B {B.shape} -> {C1.shape}")
print(f"与 matmul 一致: {np.allclose(C1, C2)}")

print(f"转置一致: {np.allclose(np.einsum('ij->ji', A), A.T)}")
print(f"迹一致: {np.allclose(np.einsum('ii->', A[:2, :2]), np.trace(A[:2, :2]))}")
print(f"外积一致: {np.allclose(np.einsum('i,j->ij', np.arange(3), np.arange(2)), np.outer(np.arange(3), np.arange(2)))}")
print(f"行和一致: {np.allclose(np.einsum('ij->i', A), A.sum(axis=1))}")


# ---------------------------------------------------------------- 2
section("2. 批量矩阵乘")
X = rng.normal(size=(5, 2, 3))
Y = rng.normal(size=(5, 3, 4))
Z1 = np.einsum("bik,bkj->bij", X, Y)
Z2 = X @ Y
print(f"einsum {Z1.shape}, matmul {Z2.shape}, 一致: {np.allclose(Z1, Z2)}")


# ---------------------------------------------------------------- 3
section("3. 多头 attention 的 einsum")
B_, H, T, d = 2, 4, 8, 4
Q = rng.normal(size=(B_, H, T, d))
K = rng.normal(size=(B_, H, T, d))
V = rng.normal(size=(B_, H, T, d))

S = np.einsum("bhid,bhjd->bhij", Q, K) / np.sqrt(d)
print(f"Q,K {Q.shape} -> S {S.shape}")

# 与 reshape/matmul 对照
S2 = (Q.reshape(B_ * H, T, d) @ K.reshape(B_ * H, T, d).transpose(0, 2, 1)).reshape(B_, H, T, T) / np.sqrt(d)
print(f"与 matmul 一致: {np.allclose(S, S2)}")

mask = np.triu(np.ones((T, T), dtype=bool), k=1)
A = softmax(np.where(mask, -1e9, S), axis=-1)
O = np.einsum("bhij,bhjd->bhid", A, V)
print(f"A {A.shape}, O {O.shape}, 行和: {A[0, 0, 0].sum():.4f}")
print(f"mask 外为 0: {np.allclose(A[0, 0][mask], 0)}")


# ---------------------------------------------------------------- 4
section("4. 随机矩阵的谱范数")
n = 500
for scale in [1 / np.sqrt(n), 1.0]:
    W = rng.normal(size=(n, n)) * scale
    s = np.linalg.svd(W, compute_uv=False)
    print(f"scale={scale:.5f}: sigma_max={s[0]:.4f}, sigma_min={s[-1]:.4f}, "
          f"cond={s[0]/s[-1]:.1f}")
print(f"理论：方差 1 时 sigma_max ≈ 2*sqrt(n) = {2*np.sqrt(n):.2f}")
print(f"      方差 1/n 时 sigma_max ≈ 2")


# ---------------------------------------------------------------- 5
section("5. Marchenko–Pastur 区间（模拟）")
n, p = 800, 400
X = rng.normal(size=(n, p))
ev = np.linalg.eigvalsh(X.T @ X / n)
gamma = p / n
lo = (1 - np.sqrt(gamma)) ** 2
hi = (1 + np.sqrt(gamma)) ** 2
print(f"gamma = p/n = {gamma}")
print(f"理论区间: [{lo:.4f}, {hi:.4f}]")
print(f"模拟最小/最大: [{ev.min():.4f}, {ev.max():.4f}]")
print(f"1%/99% 分位: {np.quantile(ev, 0.01):.4f}, {np.quantile(ev, 0.99):.4f}")
print("大部分特征值落在理论区间内；尾部有波动。")


# ---------------------------------------------------------------- 6
section("6. 小结")
print("1. einsum 用轴名统一矩阵乘/转置/迹/外积/attention。")
print("2. 多头 attention：'bhid,bhjd->bhij' 与 'bhij,bhjd->bhid'。")
print("3. 随机矩阵谱范数 ≈ 2*sqrt(n)（方差 1）；方差 1/n 时 ≈2。")
print("4. 初始化要控制谱，而不是只看元素方差。")
