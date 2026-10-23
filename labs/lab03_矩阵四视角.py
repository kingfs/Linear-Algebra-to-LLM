"""Lab03 · 矩阵四视角

对应章节：3.1 矩阵四身份 / 3.2 矩阵乘向量 / 3.3 矩阵乘法四视角

验证五件事：
1. 矩阵乘向量的列组合 = 行点积 = NumPy 的 A @ x；
2. 矩阵乘法的四种视角（行乘列、列组合、行组合、外积和）结果一致；
3. 复合映射满足 (BA)x = B(Ax)，即结合律；
4. 矩阵乘法一般不可交换；
5. FLOPs 计数 = 2mnp，并估算一个大模型线性层的计算量。

运行：uv run --with numpy python labs/lab03_矩阵四视角.py
依赖：numpy
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def matvec_by_columns(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """列组合：A x = sum_j x_j * A[:, j]。"""
    out = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        out = out + x[j] * A[:, j]
    return out


def matvec_by_rows(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """行点积：第 i 个输出是 A 的第 i 行与 x 的点积。"""
    out = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        out[i] = np.dot(A[i, :], x)
    return out


def matmul_by_rows_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """行乘列：逐元素定义，返回结果与乘加次数。"""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.zeros((m, p))
    ops = 0
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
                ops += 2  # 一次乘法 + 一次加法
    return C, ops


def matmul_by_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """列组合：C 的第 j 列 = A @ B[:, j]。"""
    m, p = A.shape[0], B.shape[1]
    C = np.zeros((m, p))
    for j in range(p):
        C[:, j] = A @ B[:, j]
    return C


def matmul_by_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """行组合：C 的第 i 行 = A[i, :] @ B。"""
    m, p = A.shape[0], B.shape[1]
    C = np.zeros((m, p))
    for i in range(m):
        C[i, :] = A[i, :] @ B
    return C


def matmul_by_outer(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """外积和：C = sum_k A[:, k] B[k, :]。"""
    m, n = A.shape
    p = B.shape[1]
    C = np.zeros((m, p))
    for k in range(n):
        C += np.outer(A[:, k], B[k, :])
    return C


# ---------------------------------------------------------------- 1
section("1. 矩阵乘向量：列组合 = 行点积 = A @ x")
A = np.array([[2.0, 1.0], [1.0, 3.0]])
x = np.array([1.0, 2.0])
by_cols = matvec_by_columns(A, x)
by_rows = matvec_by_rows(A, x)
by_numpy = A @ x
print(f"A =\n{A}\nx = {x}")
print(f"列组合 = {by_cols}")
print(f"行点积 = {by_rows}")
print(f"NumPy  = {by_numpy}")
assert np.allclose(by_cols, by_rows) and np.allclose(by_rows, by_numpy)
print("结论：三种算法得到同一个向量 (4, 7)，列组合与行点积只是求和顺序不同。")

# 随机形状再验证一次，避免只在小例子上巧合
A_rand = rng.normal(size=(4, 6))
x_rand = rng.normal(size=6)
assert np.allclose(matvec_by_columns(A_rand, x_rand), A_rand @ x_rand)
assert np.allclose(matvec_by_rows(A_rand, x_rand), A_rand @ x_rand)
print("随机 4×6 矩阵也通过：列组合 = 行点积 = A @ x。")


# ---------------------------------------------------------------- 2
section("2. 矩阵乘法四种视角：同一个 2×3 乘 3×2")
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
C_rows_cols, ops = matmul_by_rows_cols(A, B)
C_cols = matmul_by_columns(A, B)
C_rows = matmul_by_rows(A, B)
C_outer = matmul_by_outer(A, B)
C_numpy = A @ B
print(f"A =\n{A}\nB =\n{B}")
print(f"行乘列 =\n{C_rows_cols}")
print(f"列组合 =\n{C_cols}")
print(f"行组合 =\n{C_rows}")
print(f"外积和 =\n{C_outer}")
print(f"NumPy  =\n{C_numpy}")
for name, C in [("列组合", C_cols), ("行组合", C_rows), ("外积和", C_outer), ("NumPy", C_numpy)]:
    assert np.allclose(C, C_rows_cols), f"{name} 与行乘列不一致"
print("结论：四种视角结果完全一致，都是 [[4, 5], [10, 11]]。")


# ---------------------------------------------------------------- 3
section("3. 复合映射：(BA)x = B(Ax)")
B2 = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])  # 3×2
x2 = np.array([1.0, 1.0, 1.0])                        # 3 维
left = (B2 @ A) @ x2   # 先复合矩阵，再作用
right = B2 @ (A @ x2)  # 先作用 A，再作用 B
print(f"B =\n{B2}")
print(f"BA =\n{B2 @ A}")
print(f"(BA)x = {left}")
print(f"B(Ax) = {right}")
assert np.allclose(left, right)
print("结论：(BA)x = B(Ax)，矩阵乘法就是复合映射；结合律保证两种算序一致。")

# 随机形状：3×4 与 4×3
P = rng.normal(size=(3, 4))
Q = rng.normal(size=(4, 3))
z = rng.normal(size=3)
assert np.allclose((P @ Q) @ z, P @ (Q @ z))
print("随机 3×4 与 4×3 也通过：先复合与先作用结果一致。")


# ---------------------------------------------------------------- 4
section("4. 不可交换：PQ 与 QP")
P = np.array([[2.0, 1.0], [1.0, 1.0]])
Q = np.array([[1.0, 3.0], [2.0, 1.0]])
PQ = P @ Q
QP = Q @ P
print(f"P =\n{P}\nQ =\n{Q}")
print(f"PQ =\n{PQ}")
print(f"QP =\n{QP}")
assert not np.allclose(PQ, QP)
print("结论：PQ ≠ QP。矩阵乘法表示复合顺序，先 P 后 Q 与先 Q 后 P 一般不同。")
print("注意：两者形状相同不代表可以交换；交换律不成立。")

# ---------------------------------------------------------------- 5
section("5. FLOPs 计数：2mnp")
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2×3
B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3×2
C, ops = matmul_by_rows_cols(A, B)
m, n = A.shape
p = B.shape[1]
formula = 2 * m * n * p
print(f"形状：({m}×{n}) @ ({n}×{p}) = ({m}×{p})")
print(f"循环计数 = {ops} 次浮点运算")
print(f"公式 2mnp = {formula}")
assert ops == formula
print("结论：每个输出元素做 n 次乘加，共 mnp 次乘加 = 2mnp FLOPs。")

# 大模型线性层估算
Bsz, T, D, Dout = 8, 512, 4096, 11008
flops = 2 * Bsz * T * D * Dout
print(f"\n大模型线性层：B={Bsz}, T={T}, D={D}, D_out={Dout}")
print(f"FLOPs = 2·B·T·D·D_out = {flops:,} ≈ {flops / 1e9:.1f} GFLOP")
print(f"参数量 = D·D_out = {D * Dout:,} ≈ {D * Dout / 1e6:.1f}M")
print("结论：算力随 batch、序列长度与两个维度线性增长；宽度翻倍，FLOPs 变 4 倍。")

# ---------------------------------------------------------------- 6
section("6. 小结")
print("1. A x 的列组合与行点积是同一个求和，NumPy 的 @ 与它们一致。")
print("2. 矩阵乘法四种视角结果一致，选哪种取决于你想按行还是按列思考。")
print("3. (BA)x = B(Ax)：矩阵乘法是复合映射，结合律成立。")
print("4. PQ ≠ QP：复合顺序不能随意交换。")
print("5. FLOPs = 2mnp；这是估算训练/推理算力的基本公式。")

print("\n思考题：")
print("- 把第 2 节的 B 换成随机 3×2 矩阵，四种视角还会一致吗？为什么？")
print("- 若 A 是 4096×4096、B 是 4096×8，先算 AB 还是先算 BA 更省？用 2mnp 比较。")
