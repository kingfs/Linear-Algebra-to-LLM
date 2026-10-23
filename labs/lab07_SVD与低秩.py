"""Lab07 · SVD 与低秩近似

对应章节：7.1 SVD 的几何 / 7.2 计算与性质 / 7.3 低秩近似 / 7.4 伪逆与数值秩

验证：
1. 手算 SVD（由 AᵀA 的特征分解）与 np.linalg.svd 一致（允许符号自由度）；
2. A = UΣVᵀ 的精确重建；
3. 奇异值 = sqrt(λ(AᵀA))，且 σ(Aᵀ) = σ(A)；
4. 截断 SVD 的误差随 k 单调下降，并等于 sqrt(Σ_{i>k} σ_i²)；
5. Eckart–Young 最优性：随机秩 k 矩阵都赢不了 A_k；
6. 伪逆统一超定最小二乘与欠定最小范数解。

运行：
    uv run --with numpy python labs/lab07_SVD与低秩.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(7)


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def align_signs(U_hand: np.ndarray, V_hand: np.ndarray,
                U_ref: np.ndarray, V_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """奇异向量有符号自由度：逐列对齐到参考解的符号。"""
    U, V = U_hand.copy(), V_hand.copy()
    for i in range(U.shape[1]):
        if np.dot(U[:, i], U_ref[:, i]) < 0:
            U[:, i] *= -1
            V[:, i] *= -1
    return U, V


# ---------------------------------------------------------------- 1
section("1. 手算 SVD vs np.linalg.svd")
A = np.array([[3.0, 0.0], [4.0, 5.0]])
G = A.T @ A
print(f"A =\n{A}")
print(f"AᵀA =\n{G}")

lam, V = np.linalg.eigh(G)          # 对称矩阵，eigh 给升序
order = np.argsort(lam)[::-1]       # 改成降序
lam, V = lam[order], V[:, order]
sigma = np.sqrt(np.clip(lam, 0, None))
U_hand = A @ V / sigma              # u_i = A v_i / σ_i

U_np, s_np, Vt_np = np.linalg.svd(A)
V_np = Vt_np.T
U_h, V_h = align_signs(U_hand, V, U_np, V_np)

print(f"\n手算奇异值      = {sigma}")
print(f"numpy 奇异值    = {s_np}")
print(f"奇异值最大偏差  = {np.abs(sigma - s_np).max():.2e}")

print(f"\n手算 U（对齐符号后）=\n{U_h}")
print(f"numpy U             =\n{U_np}")
print(f"U 最大偏差 = {np.abs(U_h - U_np).max():.2e}")
print(f"手算 V（对齐符号后）=\n{V_h}")
print(f"numpy V             =\n{V_np}")
print(f"V 最大偏差 = {np.abs(V_h - V_np).max():.2e}")
print("（符号自由度：u_i 与 v_i 同时变号，A 不变）")


# ---------------------------------------------------------------- 2
section("2. A = UΣVᵀ 重建")
Sigma = np.diag(sigma)
A_rec = U_h @ Sigma @ V_h.T
print(f"UΣVᵀ =\n{A_rec}")
print(f"重建误差 ‖A - UΣVᵀ‖_F = {np.linalg.norm(A - A_rec):.2e}")
print(f"det(A) = {np.linalg.det(A):.6f}，σ₁σ₂ = {sigma[0] * sigma[1]:.6f}")

# 瘦 SVD 的秩 1 展开
A_1 = sigma[0] * np.outer(U_h[:, 0], V_h[:, 0])
print(f"\n秩 1 项 σ₁u₁v₁ᵀ =\n{A_1}")
print(f"这是 7.3 的 A₁；剩余项 σ₂u₂v₂ᵀ =\n{A - A_1}")


# ---------------------------------------------------------------- 3
section("3. 奇异值 = sqrt(λ(AᵀA))，且 σ(Aᵀ) = σ(A)")
for name, M in [("3×2", np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])),
                ("5×3", rng.normal(size=(5, 3)))]:
    s = np.linalg.svd(M, compute_uv=False)
    lam = np.linalg.eigvalsh(M.T @ M)[::-1]
    print(f"\n{name} 矩阵：")
    print(f"  σ                 = {s}")
    print(f"  sqrt(λ(AᵀA))      = {np.sqrt(np.clip(lam, 0, None))}")
    print(f"  最大偏差          = {np.abs(s - np.sqrt(np.clip(lam, 0, None))).max():.2e}")
    print(f"  σ(Aᵀ) 最大偏差    = {np.abs(s - np.linalg.svd(M.T, compute_uv=False)).max():.2e}")
    print(f"  ‖A‖_F² = {(M ** 2).sum():.6f}，Σσ² = {(s ** 2).sum():.6f}")


# ---------------------------------------------------------------- 4
section("4. 截断 SVD：误差随 k 下降")
m, n = 12, 8
Q1, _ = np.linalg.qr(rng.normal(size=(m, m)))
Q2, _ = np.linalg.qr(rng.normal(size=(n, n)))
sig_true = 2.0 ** (-np.arange(n))            # 快速衰减的谱：1, 0.5, 0.25, ...
A = Q1[:, :n] @ np.diag(sig_true) @ Q2.T
print(f"A 的形状 {A.shape}，谱 = {sig_true}")
print(f"{'k':>2} {'‖A-A_k‖_F':>12} {'理论值':>12} {'‖A-A_k‖₂':>12} {'理论 σ_{k+1}':>12} {'能量比':>8}")

U, s, Vt = np.linalg.svd(A, full_matrices=False)
total = (s ** 2).sum()
for k in range(0, n, 1):
    A_k = (U[:, :k] * s[:k]) @ Vt[:k]
    err_f = np.linalg.norm(A - A_k, "fro")
    err_2 = np.linalg.norm(A - A_k, 2)
    theory_f = np.sqrt((s[k:] ** 2).sum())
    theory_2 = s[k] if k < n else 0.0
    energy = (s[:k] ** 2).sum() / total
    print(f"{k:>2} {err_f:>12.6f} {theory_f:>12.6f} {err_2:>12.6f} {theory_2:>12.6f} {energy:>8.4f}")

# 误差是否单调下降
errs = [np.linalg.norm(A - (U[:, :k] * s[:k]) @ Vt[:k], "fro") for k in range(n + 1)]
print(f"\n误差单调不增？{all(errs[i] >= errs[i + 1] - 1e-12 for i in range(n))}")


# ---------------------------------------------------------------- 5
section("5. Eckart–Young：A_k 是秩 k 下最优")
print("两类「竞争者」，各随机抽样，看能否赢过 A_k：")
print("  (a) 同范数的随机秩 k 矩阵")
print("  (b) 随机 k 维子空间上的投影 P(PᵀAQ)Qᵀ")
for k in [1, 2, 3]:
    A_k = (U[:, :k] * s[:k]) @ Vt[:k]
    err_Ak = np.linalg.norm(A - A_k, "fro")
    target = np.linalg.norm(A_k, "fro")

    best_a = np.inf
    for _ in range(2000):
        G1 = rng.normal(size=(m, k))
        G2 = rng.normal(size=(n, k))
        B = G1 @ G2.T
        B = B / np.linalg.norm(B, "fro") * target      # 缩放到同一量级
        best_a = min(best_a, np.linalg.norm(A - B, "fro"))

    best_b = np.inf
    for _ in range(500):
        P, _ = np.linalg.qr(rng.normal(size=(m, k)))
        Qr, _ = np.linalg.qr(rng.normal(size=(n, k)))
        B = P @ (P.T @ A @ Qr) @ Qr.T                  # 秩 ≤ k
        best_b = min(best_b, np.linalg.norm(A - B, "fro"))

    print(f"  k={k}: ‖A-A_k‖_F = {err_Ak:.6f}")
    print(f"       （a）最好成绩 {best_a:.6f}  ≥ 上者？{best_a >= err_Ak - 1e-9}")
    print(f"       （b）最好成绩 {best_b:.6f}  ≥ 上者？{best_b >= err_Ak - 1e-9}")
print("结论：抽样再多也只能逼近，永远无法低于 A_k——这是定理，不是运气。")


# ---------------------------------------------------------------- 6
section("6. 伪逆：超定最小二乘 + 欠定最小范数")

# (a) 超定：3×2
A_over = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b_over = np.array([1.0, 1.0, 1.0])
x_pinv = np.linalg.pinv(A_over) @ b_over
x_lstsq = np.linalg.lstsq(A_over, b_over, rcond=None)[0]
x_normal = np.linalg.solve(A_over.T @ A_over, A_over.T @ b_over)
res = b_over - A_over @ x_pinv
print(f"超定 A（{A_over.shape}）：")
print(f"  pinv 解     = {x_pinv}")
print(f"  lstsq 解    = {x_lstsq}")
print(f"  正规方程解  = {x_normal}")
print(f"  三者最大偏差 = {max(np.abs(x_pinv - x_lstsq).max(), np.abs(x_pinv - x_normal).max()):.2e}")
print(f"  残差 = {res}，Aᵀr = {A_over.T @ res}（应为 0）")
print(f"  ‖r‖ = {np.linalg.norm(res):.6f}")

# (b) 欠定：2×3
A_under = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
b_under = np.array([1.0, 1.0])
x_min = np.linalg.pinv(A_under) @ b_under
null = np.linalg.svd(A_under, full_matrices=True)[2][2]   # 零空间方向
print(f"\n欠定 A（{A_under.shape}）：")
print(f"  最小范数解 = {x_min}，‖x‖ = {np.linalg.norm(x_min):.6f}")
print(f"  A @ x      = {A_under @ x_min}（应等于 b = {b_under}）")
print(f"  零空间方向 = {null}")
print("  沿零空间平移仍是解，范数严格变大：")
for t in [0.5, 1.0, 2.0]:
    x_alt = x_min + t * null
    print(f"    t={t:>3}: x = {x_alt}，‖x‖ = {np.linalg.norm(x_alt):.6f}，"
          f"残差 = {np.linalg.norm(A_under @ x_alt - b_under):.1e}")

# (c) 数值秩：小奇异值不要反演
print("\n病态例子：")
eps = 1e-8
A_ill = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
b_ill = np.array([2.0 + eps, 2.0 + eps])
s_ill = np.linalg.svd(A_ill, compute_uv=False)
print(f"  A = [[1,1],[1,1+1e-8]]，σ = [{s_ill[0]:.6f}, {s_ill[1]:.3e}]，κ = {s_ill[0] / s_ill[1]:.3e}")
print(f"  直接解      = {np.linalg.solve(A_ill, b_ill)}（真解应为 [1, 1]）")
print(f"  伪逆默认    = {np.linalg.pinv(A_ill) @ b_ill}")
print(f"  截断 rcond=1e-6 = {np.linalg.pinv(A_ill, rcond=1e-6) @ b_ill}")


# ---------------------------------------------------------------- 7
section("7. 小结")
print("1. 手算 SVD = 对 AᵀA 做特征分解 + 开方 + u_i = Av_i/σ_i，与 LAPACK 一致（差符号）。")
print("2. A = UΣVᵀ 精确重建；秩 1 项 σ₁u₁v₁ᵀ 就是低秩近似的第一层。")
print("3. σ_i = sqrt(λ_i(AᵀA))，且 ‖A‖_F² = Σσ_i²，σ(Aᵀ) = σ(A)。")
print("4. 截断误差 = sqrt(Σ_{i>k} σ_i²)，随 k 单调下降，k=r 时为 0。")
print("5. Eckart–Young：随机秩 k 矩阵最好也只是逼近 A_k，不会更好。")
print("6. 伪逆一个公式搞定超定/欠定：最小残差 + 最小范数；小奇异值要截断。")
print("\n思考题：把第 4 节的谱换成等比例衰减（σ_i 全相等），截断还有意义吗？")
print("        换成 1/i 的慢衰减，k=1 时能量比是多少？")
