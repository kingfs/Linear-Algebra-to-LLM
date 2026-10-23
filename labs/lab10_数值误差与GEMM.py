"""Lab10 · 数值误差、条件数、GEMM 与稀疏矩阵

对应章节：10.1 浮点数与误差 / 10.2 矩阵范数与条件数 /
          10.3 BLAS 与 GEMM / 10.5 稀疏矩阵

验证五件事：
1. 灾难性抵消：相近大数相减、二次方程小根、1-cos 的稳定写法；
2. 机器精度：fp16/fp32/fp64 的 1+ε 循环，以及按位模拟的 bf16；
3. 条件数如何放大解误差：2×2 病态矩阵、Hilbert 矩阵、最坏方向扰动；
4. GEMM 不同形状的耗时与 GFLOPS：方阵、以及 M 从 1 增大的 decode 形状；
5. 稀疏矩阵：COO/CSR 三个数组、SpMV、稀疏乘稀疏（纯 NumPy，不依赖 scipy）。

运行：uv run --with numpy python labs/lab10_数值误差与GEMM.py
"""

import time

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(2024)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------- 1
section("1. 灾难性抵消：有效数字是怎么没的")

x32 = np.float32(1.0000001)
print(f"x = 1.0000001 存成 fp32 后是 {float(x32):.16f}")
print(f"x - 1 = {float(x32) - 1:.16e}，真值是 1.000e-07")
print(f"相对误差 = {abs((float(x32) - 1) - 1e-7) / 1e-7:.2%}"
      "  （输入只有 7 位，结果不到 1 位可信）")

print("\n[1-cos] fp32 直接相减 vs 半角公式 2·sin²(x/2)：")
ang = np.float32(1e-4)
naive = np.float32(1.0) - np.cos(ang)
stable = np.float32(2.0) * np.sin(ang / np.float32(2.0)) ** 2
print(f"  直接：1 - cos(1e-4) = {float(naive):.6e}")
print(f"  稳定：2 sin²(5e-5)  = {float(stable):.6e}")
print("  cos(1e-4) 舍入成 1.0，直接相减把 5e-9 整个丢掉了。")

print("\n[二次方程] x² - 1e8·x + 1 = 0，fp32 求根：")
a, b, c = np.float32(1.0), np.float32(-1e8), np.float32(1.0)
disc = b * b - np.float32(4.0) * a * c
r1 = (-b + np.sqrt(disc)) / (np.float32(2.0) * a)
r2 = (-b - np.sqrt(disc)) / (np.float32(2.0) * a)
r2_stable = c / (a * r1)
print(f"  判别式 = {float(disc):.6e}（1e16-4 在 fp32 里就是 1e16）")
print(f"  求根公式：r1 = {float(r1):.6e}, r2 = {float(r2):.6e}")
print(f"  韦达定理：r2 = c/(a·r1) = {float(r2_stable):.6e}  ← 正确的小根")
print("  直接法把 1e-8 算成了 0：两个相近大数相减，有效信息全丢。")


# ---------------------------------------------------------------- 2
section("2. 机器精度：1 + ε 何时不再等于 1")

def empirical_eps(dtype):
    """循环折半，返回使 1+ε != 1 的最小 ε。"""
    one, eps = dtype(1.0), dtype(1.0)
    while one + eps / dtype(2.0) != one:
        eps /= dtype(2.0)
    return eps


for dt in (np.float16, np.float32, np.float64):
    got = empirical_eps(dt)
    print(f"{dt.__name__:>8}: 实测 ε = {float(got):.6e}, "
          f"finfo.eps = {float(np.finfo(dt).eps):.6e}")

print("\nbf16 没有内置类型，用 fp32 按位截断模拟：")
def to_bf16(x):
    """把 fp32 舍入到 bf16（7 位尾数，round-to-nearest-even）。"""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u & 0xFFFF0000).view(np.float32)

for k in (7, 8, 9):
    v = np.float32(1.0) + np.float32(2.0) ** np.float32(-k)
    print(f"  1 + 2^-{k} 在 bf16 里 = {float(to_bf16(v)):.10f}")

bf16_eps = np.float32(1.0)
while float(to_bf16(np.float32(1.0) + bf16_eps / np.float32(2.0))) != 1.0:
    bf16_eps = bf16_eps / np.float32(2.0)
print(f"  bf16 实测 ε = {float(bf16_eps):.6e}（理论 2^-7 = 7.812500e-03）")

print("\n格式对照：")
for name, eps, digits in [
    ("fp16", 2.0 ** -10, 3), ("bf16", 2.0 ** -7, 2),
    ("fp32", 2.0 ** -23, 7), ("fp64", 2.0 ** -52, 16),
]:
    print(f"  {name:>4}: ε = {eps:.3e}, 约 {digits} 位十进制有效数字")


# ---------------------------------------------------------------- 3
section("3. 条件数：同一个解，为什么误差差这么多")

def rel(x):
    n = np.linalg.norm(x)
    return n if n > 0 else 1.0

print("[3.1] A = [[1,1],[1,1+ε]]，右端相对扰动 1e-6：")
print(f"{'ε':>8}{'κ₂(A)':>12}{'相对扰动':>12}{'实际相对误差':>15}"
      f"{'κ·相对扰动':>14}")
for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    A = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
    xtrue = np.array([1.0, 1.0])
    bb = A @ xtrue
    db = rng.standard_normal(2)
    db *= 1e-6 * np.linalg.norm(bb) / np.linalg.norm(db)
    x0 = np.linalg.solve(A, bb)
    x1 = np.linalg.solve(A, bb + db)
    rel_b = np.linalg.norm(db) / np.linalg.norm(bb)
    rel_x = np.linalg.norm(x1 - x0) / rel(x0)
    kap = np.linalg.cond(A)
    print(f"{eps:>8.0e}{kap:>12.1f}{rel_b:>12.2e}{rel_x:>15.2e}"
          f"{kap * rel_b:>14.2e}")

print("\n[3.2] 把扰动放到最坏方向（SVD 的 u_max / u_min）：")
A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-4]])
U, s, Vt = np.linalg.svd(A)
bb = U[:, 0]                                  # 解主要由 v_max 贡献
xtrue = Vt.T[:, 0] / s[0]
db = (1e-6 * np.linalg.norm(bb)) * U[:, -1]    # 误差放在最敏感方向
x0 = np.linalg.solve(A, bb)
x1 = np.linalg.solve(A, bb + db)
rel_b = np.linalg.norm(db) / np.linalg.norm(bb)
rel_x = np.linalg.norm(x1 - x0) / rel(x0)
print(f"  σ_max={s[0]:.4e}, σ_min={s[-1]:.4e}, κ={s[0] / s[-1]:.1f}")
print(f"  相对扰动={rel_b:.2e}, 实际相对误差={rel_x:.2e}, "
      f"κ·相对扰动={np.linalg.cond(A) * rel_b:.2e}")
print("  实际误差几乎顶到上界：这个扰动正好落在最敏感方向。")

print("\n[3.3] Hilbert 矩阵 H_n：条件数随 n 爆炸")
def hilbert(n):
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])

print(f"{'n':>4}{'κ₂(H_n)':>16}{'解相对误差':>16}")
for n in (4, 5, 6, 7, 8):
    H = hilbert(n)
    xtrue = np.ones(n)
    bb = H @ xtrue
    db = rng.standard_normal(n)
    db *= 1e-8 * np.linalg.norm(bb) / np.linalg.norm(db)
    x1 = np.linalg.solve(H, bb + db)
    err = np.linalg.norm(x1 - xtrue) / np.linalg.norm(xtrue)
    print(f"{n:>4}{np.linalg.cond(H):>16.3e}{err:>16.3e}")
print("  n=8 时 κ 已到 1e10，双精度只剩约 6 位有效数字。")

print("\n[3.4] 正交矩阵作对照（κ=1）：")
Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
bb = Q @ np.ones(4)
db = rng.standard_normal(4)
db *= 1e-6 * np.linalg.norm(bb) / np.linalg.norm(db)
print(f"  κ(Q) = {np.linalg.cond(Q):.3f}, "
      f"相对误差 = {np.linalg.norm(np.linalg.solve(Q, bb + db) - np.ones(4)) / 2:.2e}"
      "（≈ 相对扰动本身）")


# ---------------------------------------------------------------- 4
section("4. GEMM：形状如何决定吞吐")

def bench_gemm(M, N, K, reps=3):
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = np.empty((M, N), dtype=np.float32)
    np.matmul(A, B, out=C)                     # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        np.matmul(A, B, out=C)
    dt = (time.perf_counter() - t0) / reps
    flops = 2.0 * M * N * K
    nbytes = 4.0 * (M * K + K * N + M * N)
    return dt, flops, nbytes

print("[4.1] 方阵 n×n×n：FLOPs 按 n³ 涨，时间也应约 8 倍/翻倍")
print(f"{'n':>6}{'时间(ms)':>12}{'GFLOPS':>12}{'AI':>10}")
for n in (128, 256, 512, 1024):
    dt, flops, nbytes = bench_gemm(n, n, n)
    print(f"{n:>6}{dt * 1e3:>12.3f}{flops / dt / 1e9:>12.1f}"
          f"{flops / nbytes:>10.1f}")

print("\n[4.2] 固定 N=K=2048，改变 M：M=1 是 decode，M 大是 prefill")
print(f"{'M':>6}{'时间(ms)':>12}{'GFLOPS':>12}{'AI':>10}")
for M in (1, 8, 64, 256, 1024):
    dt, flops, nbytes = bench_gemm(M, 2048, 2048)
    print(f"{M:>6}{dt * 1e3:>12.3f}{flops / dt / 1e9:>12.1f}"
          f"{flops / nbytes:>10.1f}")
print("  M=1 时 AI≈0.5（fp32），GFLOPS 很低：权重只读一遍、每个数只做一次乘加；")
print("  M 增大后同一份权重被复用 M 次，AI 上升，GFLOPS 才爬向峰值。")


# ---------------------------------------------------------------- 5
section("5. 稀疏矩阵：只存非零元、只算非零元")

def build_sparse(n, density):
    """构造随机稀疏矩阵，返回 COO 与 CSR（纯 NumPy）。"""
    m = int(n * n * density)
    rows = rng.integers(0, n, m, dtype=np.int64)
    cols = rng.integers(0, n, m, dtype=np.int64)
    vals = rng.standard_normal(m).astype(np.float32)
    key = rows * n + cols
    uniq, inv = np.unique(key, return_inverse=True)
    data = np.zeros(len(uniq), dtype=np.float32)
    np.add.at(data, inv, vals)                 # 合并重复坐标
    keep = data != 0
    uniq, data = uniq[keep], data[keep]
    rows, cols = uniq // n, uniq % n           # 已按 (row, col) 排序
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(rows, minlength=n))
    return rows, cols, data, indptr


def spmv_coo(rows, cols, data, x, n):
    return np.bincount(rows, weights=data * x[cols], minlength=n).astype(np.float32)


def spmv_csr(indptr, indices, data, x):
    """按行压缩：用 indptr 生成每个非零元的行号，再按行求和。"""
    n = len(indptr) - 1
    row_ids = np.repeat(np.arange(n), np.diff(indptr))
    contrib = data * x[indices]
    return np.bincount(row_ids, weights=contrib, minlength=n).astype(np.float32)


n = 4000
rows, cols, data, indptr = build_sparse(n, 0.001)
indices = cols.astype(np.int32)
nnz = len(data)
print(f"矩阵 {n}×{n}，nnz = {nnz}，密度 = {nnz / (n * n):.5%}，"
      f"稀疏度 = {1 - nnz / (n * n):.5%}")
print(f"稠密存储 = {n * n * 4 / 1e6:.1f} MB；"
      f"CSR = {(data.nbytes + indices.nbytes + indptr.nbytes) / 1e6:.3f} MB；"
      f"COO = {(data.nbytes + rows.nbytes + cols.nbytes) / 1e6:.3f} MB")

xs = rng.standard_normal(n).astype(np.float32)
dense = np.zeros((n, n), dtype=np.float32)
dense[rows, cols] = data
y_ref = dense @ xs
y_coo = spmv_coo(rows, cols, data, xs, n)
y_csr = spmv_csr(indptr, indices, data, xs)

print(f"\nSpMV 误差：COO {np.linalg.norm(y_coo - y_ref):.2e}，"
      f"CSR {np.linalg.norm(y_csr - y_ref):.2e}")

t0 = time.perf_counter()
for _ in range(20):
    spmv_csr(indptr, indices, data, xs)
t_csr = (time.perf_counter() - t0) / 20
t0 = time.perf_counter()
for _ in range(20):
    dense @ xs
t_dense = (time.perf_counter() - t0) / 20
print(f"单次耗时：CSR {t_csr * 1e6:.1f} µs，稠密 {t_dense * 1e6:.1f} µs"
      f"（稀疏格式有索引开销）")

print("\n[5.2] 稀疏乘稀疏 C = A·B（按 A 的每个非零元累加 B 的整行）：")
m = 200
ra, ca, da, ia = build_sparse(m, 0.02)
rb, cb, db, ib = build_sparse(m, 0.02)


def csr_matmul(ia, ca, da, ib, cb, db, n):
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for p in range(ia[i], ia[i + 1]):
            k = ca[p]
            s, e = ib[k], ib[k + 1]
            C[i, cb[s:e]] += da[p] * db[s:e]
    return C


Ad = np.zeros((m, m), dtype=np.float32); Ad[ra, ca] = da
Bd = np.zeros((m, m), dtype=np.float32); Bd[rb, cb] = db
C = csr_matmul(ia, ca, da, ib, cb, db, m)
print(f"  nnz(A)={len(da)}, nnz(B)={len(db)}, nnz(C)={np.count_nonzero(C)}")
print(f"  与稠密乘法最大差异 = {np.abs(C - Ad @ Bd).max():.2e}")

print("\n[5.3] 4×4 小例子（与 10.5 节相同）：")
A4 = np.array([[1, 0, 0, 2],
               [0, 3, 0, 0],
               [0, 0, 0, 0],
               [4, 0, 5, 0]], dtype=np.float32)
r4, c4 = np.nonzero(A4)
v4 = A4[r4, c4]
ip4 = np.zeros(5, dtype=np.int64)
ip4[1:] = np.cumsum(np.bincount(r4, minlength=4))
print(f"  values  = {v4}")
print(f"  col_idx = {c4}")
print(f"  indptr  = {ip4}")
print(f"  A·(1,2,3,4) = {spmv_csr(ip4, c4.astype(np.int32), v4, np.array([1, 2, 3, 4], dtype=np.float32))}")

# ---------------------------------------------------------------- 6
section("6. 小结")
print("1. 相近大数相减会把绝对误差放大成巨大的相对误差；换公式（韦达/半角）比换精度更有效。")
print("2. 机器精度 ε 是 1 附近的间距：fp16≈9.77e-4，bf16≈7.8e-3，fp32≈1.19e-7。")
print("3. 条件数 κ 是误差放大上界；κ 大时任何算法都救不了，只能预处理或换问题。")
print("4. GEMM 的吞吐由算术强度决定；M=1 的 decode 是访存受限，增大 batch 才能复用权重。")
print("5. 稀疏格式省掉零，但多了索引；SpMV 约 O(nnz)，稀疏乘稀疏约 O(nnz·d) 且可能填充。")

print("\n思考题：")
print("- 把第 1 节的 1e-4 改成 1e-2，1-cos 的直接算法还准吗？为什么？")
print("- 第 3.1 节里 ε=1e-10 时，κ·相对扰动已经大于 1；这时的解还有意义吗？")
print("- 第 4.2 节把 N、K 固定为 2048，若改成 4096，M 的拐点会怎么移动？")
print("- 第 5 节把密度从 0.005 提到 0.5，CSR 还会比稠密快吗？动手改一下。")
