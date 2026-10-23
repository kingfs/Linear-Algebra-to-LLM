"""Lab06 · 特征值与幂迭代

对应章节：6.1 特征值 / 6.2 特征多项式 / 6.3 对角化 / 6.4 谱定理 /
          6.5 二次型 / 6.6 幂迭代与 PageRank / 6.7 Jordan 与矩阵指数

验证：
1. 2×2、3×3 特征值与手算对照；
2. A = P D P^{-1} 的重建与 A^k；
3. 对称矩阵的正交对角化 A = Q Λ Q^T；
4. 二次型的正定判据与 Cholesky 分解；
5. 幂迭代的收敛速度由 |λ2/λ1| 决定；
6. 三节点 PageRank 的迭代与平稳方程；
7. Jordan 块的矩阵指数与瞬态增长。

运行：uv run --with numpy python labs/lab06_特征值与幂迭代.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)
np.random.seed(7)


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def check(name: str, ok: bool) -> None:
    print(f"[{'通过' if ok else '失败'}] {name}")


def expm_taylor(M: np.ndarray, terms: int = 30) -> np.ndarray:
    """缩放-平方 + Taylor 级数计算矩阵指数。"""
    norm = np.linalg.norm(M, 2)
    s = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    A = M / (2 ** s)
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, terms + 1):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


# ---------------------------------------------------------------- 1
section("1. 2×2 手算对照：A = [[2, 1], [1, 2]]")
A = np.array([[2.0, 1.0], [1.0, 2.0]])
w, V = np.linalg.eigh(A)
print(f"A =\n{A}")
print(f"trace(A) = {np.trace(A):.1f}, det(A) = {np.linalg.det(A):.1f}")
print(f"手算特征值：3 和 1；NumPy：{w}")
check("迹 = 特征值之和", np.isclose(np.trace(A), w.sum()))
check("行列式 = 特征值之积", np.isclose(np.linalg.det(A), np.prod(w)))
for i in range(2):
    v = V[:, i]
    residual = np.linalg.norm(A @ v - w[i] * v)
    print(f"λ={w[i]:.0f} 的特征向量 v={v}，‖Av-λv‖={residual:.2e}")
    check(f"λ={w[i]:.0f} 的特征方程", residual < 1e-12)
print(f"A^10（NumPy）=\n{np.linalg.matrix_power(A, 10)}")
print("手算 A^10 = [[29525, 29524], [29524, 29525]]")


# ---------------------------------------------------------------- 2
section("2. 3×3 上三角矩阵：特征多项式、迹与行列式")
B = np.array([
    [2.0, 0.0, 0.0],
    [1.0, 3.0, 0.0],
    [4.0, 5.0, 6.0],
])
wb = np.linalg.eigvals(B)
print(f"B =\n{B}")
print(f"特征值（NumPy）={np.sort(wb)}；手算 2, 3, 6")
print(f"特征多项式系数（从高次到低次）={np.poly(B)}")
print("手算 p(λ)=λ^3-11λ^2+36λ-36")
print(f"trace={np.trace(B):.1f}, det={np.linalg.det(B):.1f}")
check("特征值之和 = 迹", np.isclose(wb.sum(), np.trace(B)))
check("特征值之积 = 行列式", np.isclose(np.prod(wb), np.linalg.det(B)))


# ---------------------------------------------------------------- 3
section("3. A = P D P^{-1} 重建与不可对角化")
P = np.array([[1.0, 1.0], [1.0, -1.0]])
D = np.diag([3.0, 1.0])
Pinv = np.linalg.inv(P)
print(f"P =\n{P}\nD =\n{D}\nP^{-1} =\n{Pinv}")
check("P D P^{-1} = A", np.allclose(P @ D @ Pinv, A))
print(f"P D^10 P^{-1} 的第一列：{(P @ np.linalg.matrix_power(D, 10) @ Pinv)[:, 0]}")

N = np.array([[1.0, 1.0], [0.0, 1.0]])
wn, Vn = np.linalg.eig(N)
print(f"\n亏损矩阵 N =\n{N}")
print(f"特征值={wn}，代数重数均为 2")
print(f"rank(N-I)={np.linalg.matrix_rank(N - np.eye(2))}，几何重数={2 - np.linalg.matrix_rank(N - np.eye(2))}")
check("几何重数 1 < 代数重数 2", 2 - np.linalg.matrix_rank(N - np.eye(2)) == 1)
print("结论：只有一个特征方向，P 不可逆，不能对角化（见 6.7 的 Jordan 块）。")


# ---------------------------------------------------------------- 4
section("4. 对称矩阵的正交对角化：A = Q Λ Q^T")
S = np.array([[2.0, 2.0], [2.0, -1.0]])
ws, Q = np.linalg.eigh(S)
print(f"S =\n{S}")
print(f"特征值={ws}，特征向量（列）=\n{Q}")
check("Q^T Q = I", np.allclose(Q.T @ Q, np.eye(2)))
check("Q^T S Q = diag(λ)", np.allclose(Q.T @ S @ Q, np.diag(ws)))
check("Q Λ Q^T = S", np.allclose(Q @ np.diag(ws) @ Q.T, S))
print(f"两个特征向量正交：q1·q2 = {Q[:, 0] @ Q[:, 1]:.2e}")
print("手算 λ=3 对应 (2,1)/√5，λ=-2 对应 (1,-2)/√5")


# ---------------------------------------------------------------- 5
section("5. 二次型、正定性与 Cholesky")
Qform = np.array([[2.0, 1.0], [1.0, 2.0]])
print(f"A =\n{Qform}，Q(x) = 2x1^2 + 2x1x2 + 2x2^2")
minor1 = Qform[0, 0]
minor2 = np.linalg.det(Qform)
print(f"顺序主子式：Δ1={minor1:.1f}, Δ2={minor2:.1f}")
check("Sylvester 判据：Δ1>0 且 Δ2>0", minor1 > 0 and minor2 > 0)
L = np.linalg.cholesky(Qform)
print(f"Cholesky L =\n{L}")
check("L L^T = A", np.allclose(L @ L.T, Qform))
for x in [np.array([1.0, 0.0]), np.array([1.0, -1.0]), np.array([1.0, 1.0])]:
    print(f"Q({x}) = {x @ Qform @ x:.2f}")
indef = np.array([[1.0, 2.0], [2.0, 1.0]])
print(f"\n反例 [[1,2],[2,1]]：Δ1={indef[0, 0]:.1f}, Δ2={np.linalg.det(indef):.1f}, "
      f"特征值={np.linalg.eigvalsh(indef)}")
print("对角线为正但 Δ2<0，故不定；x=(1,-1) 时 Q=-2。")


# ---------------------------------------------------------------- 6
section("6. 幂迭代：收敛速度由 |λ2/λ1| 决定")


def power_iteration(M: np.ndarray, x0: np.ndarray, steps: int):
    x = x0 / np.linalg.norm(x0)
    history = []
    for _ in range(steps):
        y = M @ x
        x = y / np.linalg.norm(y)
        rayleigh = x @ M @ x
        history.append((x.copy(), rayleigh))
    return history


def angle_to(u: np.ndarray, v: np.ndarray) -> float:
    c = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# 情形 A：比值 1/3
print("情形 A：A=[[2,1],[1,2]]，λ1=3, λ2=1，比值 1/3")
hist = power_iteration(A, np.array([1.0, 0.0]), 12)
v_true = V[:, 1]  # λ=3 的特征向量
for k in [1, 2, 3, 5, 8, 12]:
    x, rq = hist[k - 1]
    print(f"  k={k:2d}  λ≈{rq:.8f}  与主方向夹角={angle_to(x, v_true):.3e}°")
errs = [np.linalg.norm(hist[k][0] - np.sign(hist[k][0] @ v_true) * v_true) for k in range(12)]
print(f"  相邻误差比（应趋近 1/3）：{[round(float(errs[i + 1] / errs[i]), 4) for i in range(5, 10)]}")

# 情形 B：比值 0.9，构造对称矩阵
theta = 0.4
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
slow = R @ np.diag([1.0, 0.9]) @ R.T
print("\n情形 B：对称矩阵，λ1=1, λ2=0.9，比值 0.9")
hist2 = power_iteration(slow, np.array([1.0, 0.0]), 30)
w2, V2 = np.linalg.eigh(slow)
v2_true = V2[:, 1]
for k in [5, 10, 20, 30]:
    x, rq = hist2[k - 1]
    print(f"  k={k:2d}  λ≈{rq:.8f}  与主方向夹角={angle_to(x, v2_true):.3e}°")
errs2 = [np.linalg.norm(hist2[k][0] - np.sign(hist2[k][0] @ v2_true) * v2_true) for k in range(30)]
print(f"  相邻误差比（应趋近 0.9）：{[round(float(errs2[i + 1] / errs2[i]), 4) for i in range(10, 15)]}")
print("结论：同样的 30 步，比值 1/3 早已收敛，比值 0.9 仍明显偏离主方向。")


# ---------------------------------------------------------------- 7
section("7. 三节点 PageRank：迭代与平稳方程")
Pmat = np.array([
    [0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0],
    [0.5, 1.0, 0.0],
])
d = 0.85
n = 3
G = d * Pmat + (1 - d) / n * np.ones((n, n))
print(f"转移矩阵 P（列和为 1）=\n{Pmat}")
print(f"阻尼矩阵 G =\n{G}")
pi = np.ones(n) / n
for _ in range(200):
    pi = G @ pi
print(f"幂迭代 200 步：π = {pi}")
sol = np.linalg.solve(np.eye(n) - d * Pmat, (1 - d) / n * np.ones(n))
print(f"解平稳方程：  π = {sol}")
check("幂迭代收敛到平稳解", np.allclose(pi, sol, atol=1e-10))
print(f"手算 π ≈ (0.3878, 0.2148, 0.3974)，网页 3 最高")

ev = np.linalg.eigvals(G)
ev_abs = np.sort(np.abs(ev))[::-1]
print(f"G 的特征值模（降序）={ev_abs}")
print(f"收敛比 |λ2|≈{ev_abs[1]:.4f}，与 d·|λ2(P)|=0.85×0.7071≈0.60 一致")


# ---------------------------------------------------------------- 8
section("8. Jordan 块与矩阵指数：瞬态增长")
J = np.array([[1.0, 1.0], [0.0, 1.0]])
t = 1.0
expJ = expm_taylor(J * t)
closed = np.exp(t) * np.array([[1.0, t], [0.0, 1.0]])
print(f"J = [[1,1],[0,1]]，e^J（Taylor）=\n{expJ}")
print(f"闭式 e·[[1,1],[0,1]] =\n{closed}")
check("矩阵指数与闭式一致", np.allclose(expJ, closed, atol=1e-10))

A_growth = np.array([[-1.0, 100.0], [0.0, -1.0]])
print("\nA = [[-1,100],[0,-1]]：特征值全为 -1，但可能瞬态增长")
for t in [0.0, 0.01, 0.5, 1.0, 1.5, 2.0]:
    E = expm_taylor(A_growth * t)
    print(f"  t={t:4.2f}  ‖e^(At)‖_2={np.linalg.norm(E, 2):8.4f}  e^(At) 右上角={E[0, 1]:9.4f}")
print("峰值约 36.8（t≈1），之后才指数衰减；只看特征值会漏掉这次放大。")


# ---------------------------------------------------------------- 9
section("9. 总结")
print("1) 特征值/特征向量：Av=λv，迹与行列式是谱的两个坐标。")
print("2) 对角化：A=PDP^{-1}，A^k=P D^k P^{-1}；不可对角化时用 Jordan 块。")
print("3) 对称矩阵：A=QΛQ^T，特征值实、特征向量正交。")
print("4) 正定：所有特征值>0 ⇔ 顺序主子式>0 ⇔ Cholesky 成功。")
print("5) 幂迭代收敛比 |λ2/λ1|；PageRank 用阻尼保证唯一平稳分布。")
print("6) 矩阵指数：亏损矩阵多出 t 因子，非正规矩阵可能瞬态增长。")
