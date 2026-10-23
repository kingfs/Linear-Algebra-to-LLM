"""Lab04 · 最小二乘与条件数

对应章节：4.4 最小二乘 / 4.5 伪逆 / 4.6 条件数

验证四件事：
1. 无解系统的最小二乘解 = 正规方程解 = np.linalg.lstsq 解；
2. 残差与 A 的列空间正交（Aᵀr ≈ 0），且满足勾股分解；
3. 拟合直线与原始点的位置关系（文本图）；
4. 条件数越大的矩阵，解对右端扰动越敏感，且误差接近 κ·(相对扰动)。

运行：uv run --with numpy python labs/lab04_最小二乘.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def ascii_plot(xs: np.ndarray, ys: np.ndarray,
               a: float, b: float, rows: int = 11, cols: int = 46) -> None:
    """用字符画散点与拟合直线：o 是数据点，* 是直线。"""
    x_lo, x_hi = xs.min() - 0.4, xs.max() + 0.4
    y_lo = min(ys.min(), a + b * x_lo) - 0.4
    y_hi = max(ys.max(), a + b * x_hi) + 0.4
    grid = [[" "] * cols for _ in range(rows)]

    def to_cell(x: float, y: float) -> tuple[int, int]:
        c = int((x - x_lo) / (x_hi - x_lo) * (cols - 1))
        r = int((y_lo - y) / (y_hi - y_lo) * (rows - 1)) + rows - 1
        return max(0, min(rows - 1, r)), max(0, min(cols - 1, c))

    for c in range(cols):                      # 先画直线
        x = x_lo + (x_hi - x_lo) * c / (cols - 1)
        r, cc = to_cell(x, a + b * x)
        grid[r][cc] = "*"
    for x, y in zip(xs, ys):                   # 数据点覆盖在直线上
        r, cc = to_cell(float(x), float(y))
        grid[r][cc] = "o"

    print("  y")
    for r, line in enumerate(grid):
        print(f"  {''.join(line)}")
    print(f"  +{'-' * cols}> x    （o=数据，*=拟合直线 y={a:.3f}+{b:.3f}x）")


# ---------------------------------------------------------------- 1
section("1. 无解系统的最小二乘")
# 拟合 y = a + b x，数据点 (0,0), (1,1), (2,3)，三点不共线
xs = np.array([0.0, 1.0, 2.0])
ys = np.array([0.0, 1.0, 3.0])
A = np.column_stack([np.ones_like(xs), xs])      # [3, 2]
b = ys

print(f"A =\n{A}")
print(f"b = {b}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}，未知数 2 个："
      f"{'唯一最小二乘解' if np.linalg.matrix_rank(A) == 2 else '解不唯一'}")
print("A·x = b 本身无解（三个方程、两个未知数，三点不共线）")

normal = np.linalg.solve(A.T @ A, A.T @ b)       # 正规方程
lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)    # SVD/QR
pinv = np.linalg.pinv(A) @ b                     # 伪逆

print(f"\n正规方程解 (AᵀA)⁻¹Aᵀb = {normal}  （解析值 [-1/6, 3/2] = [-0.1667, 1.5]）")
print(f"np.linalg.lstsq 解     = {lstsq}")
print(f"np.linalg.pinv 解      = {pinv}")
print(f"三者最大差异 = {np.max(np.abs(normal - lstsq)):.2e}")

# ---------------------------------------------------------------- 2
section("2. 残差与列空间正交")
pred = A @ normal
resid = b - pred
sse = float(resid @ resid)
print(f"拟合值 A·x̂ = {pred}")
print(f"残差 r = b - A·x̂ = {resid}")
print(f"Aᵀr = {A.T @ resid}   （应为 0，说明 r ⊥ col(A)）")
print(f"残差平方和 SSE = {sse:.6f}  （解析值 1/6 ≈ 0.166667）")
print(f"‖b‖² = {b @ b:.4f}，‖ŷ‖² + ‖r‖² = {pred @ pred + resid @ resid:.4f}"
      "（勾股分解）")

# ---------------------------------------------------------------- 3
section("3. 拟合直线")
print(f"直线：y = {normal[0]:.4f} + {normal[1]:.4f} x")
print(f"{'x':>6}{'y':>8}{'ŷ':>10}{'残差':>10}")
for x, y, yhat, r in zip(xs, ys, pred, resid):
    print(f"{x:>6.2f}{y:>8.2f}{yhat:>10.4f}{r:>10.4f}")
print()
ascii_plot(xs, ys, float(normal[0]), float(normal[1]))

# ---------------------------------------------------------------- 4
section("4. 条件数：同一个问题，不同敏感度")
eps_list = [1e-2, 1e-4]
delta = np.array([1e-5, -1e-5])
print(f"扰动 δb = ({delta[0]:.0e}, {delta[1]:.0e})（相对大小与 b 有关，见下表）\n")
print(f"{'ε':>8}{'κ(A)':>12}{'相对扰动':>12}{'实际相对误差':>14}{'κ·相对扰动':>14}")
for eps in eps_list:
    M = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
    rhs = np.array([2.0, 2.0 + eps])              # 精确解 (1, 1)
    x0 = np.linalg.solve(M, rhs)
    x1 = np.linalg.solve(M, rhs + delta)
    rel_x = float(np.linalg.norm(x1 - x0) / np.linalg.norm(x0))
    rel_b = float(np.linalg.norm(delta) / np.linalg.norm(rhs))
    kappa = float(np.linalg.cond(M))
    print(f"{eps:>8.0e}{kappa:>12.1f}{rel_b:>12.2e}{rel_x:>14.2e}{kappa * rel_b:>14.2e}")

print("\n理论：κ ≈ 4/ε（见 4.6 的推导）；误差不会超过 κ·(相对扰动)。")
print("正交矩阵作为对照（κ = 1）：")
Q, _ = np.linalg.qr(np.random.randn(2, 2))
x0 = np.linalg.solve(Q, np.array([1.0, 1.0]))
x1 = np.linalg.solve(Q, np.array([1.0, 1.0]) + delta)
print(f"κ(Q) = {np.linalg.cond(Q):.3f}，实际相对误差 = "
      f"{np.linalg.norm(x1 - x0) / np.linalg.norm(x0):.2e}（≈ 相对扰动本身）")

# ---------------------------------------------------------------- 5
section("5. Hilbert 矩阵：条件数增长有多快")
for n in range(2, 7):
    H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    print(f"H{n}: κ = {np.linalg.cond(H):.3e}")
print("n=6 时 κ 已到 1e7 量级：双精度下解大约只剩 9 位有效数字。")

# ---------------------------------------------------------------- 6
section("6. 小结")
print("1. 无解时最小二乘给出『最近』的解：正规方程、lstsq、伪逆三者一致。")
print("2. 最优的充要条件是残差与列空间正交：Aᵀr = 0。")
print("3. 残差平方和就是 b 到 col(A) 的距离平方。")
print("4. 条件数是问题的属性：κ 越大，同样的输入扰动被放大得越厉害。")
print("5. 正规方程的条件数是 κ(A)²，所以实践优先用 lstsq / QR。")

print("\n思考题：")
print("- 把第 1 节的数据点改成 (0,0), (1,1), (2,2)，残差是多少？A 的秩是多少？")
print("- 若给第 4 节的矩阵加上岭回归项 λI，κ 会怎么变？试一试 λ=0.01。")
print("- 第 2 节里 ‖ŷ‖² + ‖r‖² = ‖b‖²，如果换一个拟合直线，这个等式还成立吗？")
