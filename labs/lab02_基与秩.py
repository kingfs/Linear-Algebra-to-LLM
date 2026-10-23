"""Lab02 · 基、秩与 Gram–Schmidt

对应章节：2.3 线性无关 / 2.4 基与维数 / 2.5 坐标与基变换 / 2.6 秩 / 2.7 正交化

验证：
1. 秩 = 最大无关组大小 = 列空间维数；秩-零化度定理；
2. 零空间与行空间正交；
3. 坐标随基改变，向量不变；
4. Gram–Schmidt 得到的基确实标准正交；
5. 相似矩阵特征值相同；
6. QR 分解与投影。

运行：uv run --with numpy python lab02_基与秩.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
np.random.seed(7)


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """对列向量做经典 Gram–Schmidt，返回标准正交基（按列）。"""
    m, n = vectors.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = vectors[:, j].astype(float).copy()
        for i in range(j):
            v -= (Q[:, i] @ vectors[:, j]) * Q[:, i]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError("向量组线性相关，无法正交化")
        Q[:, j] = v / norm
    return Q


# ---------------------------------------------------------------- 1
section("1. 秩、零空间与秩-零化度")
A = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [1.0, 1.0, 1.0],
])
rank = np.linalg.matrix_rank(A)
print(f"A =\n{A}")
print(f"rank(A) = {rank}")
print(f"零空间维数应为 n - rank = {A.shape[1]} - {rank} = {A.shape[1] - rank}")

# 用 SVD 求零空间（右奇异向量中奇异值为 0 的部分）
U, s, Vt = np.linalg.svd(A)
null_basis = Vt[rank:].T
print(f"奇异值 = {s}")
print(f"零空间基（列）=\n{null_basis}")
print(f"A @ 零空间基 ≈\n{A @ null_basis}")

# 行空间与零空间正交
row_space = Vt[:rank].T
print(f"行空间基（列）=\n{row_space}")
print(f"行空间ᵀ @ 零空间 ≈\n{row_space.T @ null_basis}（应为 0）")


# ---------------------------------------------------------------- 2
section("2. 坐标随基改变")
B = np.array([[1.0, 1.0], [1.0, -1.0]])  # 列是基向量
v = np.array([3.0, 1.0])
coords = np.linalg.solve(B, v)
print(f"基 B 的列 = {B.tolist()}")
print(f"v = {v}")
print(f"v 在 B 下的坐标 = {coords}")
print(f"用坐标重建 v = B @ coords = {B @ coords}")

P = B  # 标准基 -> B 的基变换矩阵
print(f"\n同一个 v 在标准基下坐标 = {v}")
print(f"换基后坐标 = {coords}（不同，但 B @ coords 仍是 v）")


# ---------------------------------------------------------------- 3
section("3. Gram–Schmidt 与 QR")
V = np.array([
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
])
Q = gram_schmidt(V)
print(f"原始列向量 V =\n{V}")
print(f"Gram–Schmidt 结果 Q =\n{Q}")
print(f"QᵀQ =\n{Q.T @ Q}（应接近单位阵）")

R = Q.T @ V
print(f"R = QᵀV =\n{R}（应为上三角）")
print(f"QR 重建误差 = {np.linalg.norm(V - Q @ R):.2e}")

# 与 numpy 的 QR 比较（允许列符号不同）
Q_np, R_np = np.linalg.qr(V)
print(f"\nnumpy QR 的 Q =\n{Q_np}")
print("（与手写结果可能相差若干列的符号，这是 QR 的符号自由度）")


# ---------------------------------------------------------------- 4
section("4. 相似变换不改变特征值")
A2 = np.array([[2.0, 1.0], [0.0, 3.0]])
P2 = np.array([[1.0, 1.0], [1.0, -1.0]])
A_B = np.linalg.inv(P2) @ A2 @ P2
print(f"A（标准基）=\n{A2}")
print(f"P⁻¹AP（基 B 下）=\n{A_B}")
print(f"特征值 A   = {np.linalg.eigvals(A2)}")
print(f"特征值 P⁻¹AP = {np.linalg.eigvals(A_B)}")
print(f"迹: {np.trace(A2):.4f} vs {np.trace(A_B):.4f}")
print(f"行列式: {np.linalg.det(A2):.4f} vs {np.linalg.det(A_B):.4f}")


# ---------------------------------------------------------------- 5
section("5. 低秩：外积与近似")
u = np.array([1.0, 2.0, 3.0])
w = np.array([2.0, -1.0, 0.5])
outer = np.outer(u, w)
print(f"u wᵀ =\n{outer}")
print(f"rank(u wᵀ) = {np.linalg.matrix_rank(outer)}（应为 1）")

# 随机矩阵的低秩近似
M = np.random.randn(8, 6)
U, s, Vt = np.linalg.svd(M, full_matrices=False)
print(f"\n随机 8×6 矩阵的奇异值 = {s}")
for k in [1, 3, 5]:
    M_k = (U[:, :k] * s[:k]) @ Vt[:k]
    err = np.linalg.norm(M - M_k, "fro")
    print(f"保留秩 {k}: Frobenius 误差 = {err:.4f}")


# ---------------------------------------------------------------- 6
section("6. 小结")
print("1. 秩 = 列空间维数；零空间维数 = n - 秩。")
print("2. 行空间与零空间正交，二者直和是整个输入空间。")
print("3. 坐标随基改变，向量不变；相似矩阵特征值相同。")
print("4. Gram–Schmidt 逐列减去已有方向的投影，得到标准正交基。")
print("5. 低秩近似误差随 k 增大而下降（Eckart–Young 最优）。")
print("\n思考题：若把 Gram–Schmidt 的输入顺序换一下，得到的 Q 会变吗？张成空间会变吗？")
