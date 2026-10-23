"""Lab05 · 行列式

对应章节：5.1 体积缩放 / 5.2 性质与计算 / 5.3 逆与可逆性 / 5.4 Jacobian

验证六件事：
1. 2×2 行列式等于单位正方形像的有向面积；3×3 等于单位立方体像的体积；
2. 随机矩阵的行列式等于线性变换后的网格面积比；
3. det(AB) = det(A) det(B)，以及 det(Aᵀ)、det(A⁻¹)；
4. 奇异矩阵 det ≈ 0，近奇异矩阵 det 很小但条件数很大；
5. det 小不等于病态：条件数与 det 是两回事，高维 det 还会下溢；
6. 极坐标的 Jacobian 行列式等于 r，即 dx dy = r dr dθ。

运行：uv run --with numpy python labs/lab05_行列式.py
依赖：numpy
"""

import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def polygon_area(pts: np.ndarray) -> float:
    """鞋带公式：按顺序给出多边形顶点，返回面积（非负）。"""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def grid_area_ratio(A: np.ndarray, n: int = 6, lo: float = -1.5, hi: float = 1.5) -> float:
    """把 [lo,hi]² 切成 n×n 个小格，逐格变换后求面积和，再除以原面积。"""
    xs = np.linspace(lo, hi, n + 1)
    total = 0.0
    for i in range(n):
        for j in range(n):
            cell = np.array([
                [xs[i], xs[j]],
                [xs[i + 1], xs[j]],
                [xs[i + 1], xs[j + 1]],
                [xs[i], xs[j + 1]],
            ])
            total += polygon_area(cell @ A.T)
    return total / (hi - lo) ** 2


# ---------------------------------------------------------------- 1
section("1. 2×2：行列式 = 有向面积")
A = np.array([[3.0, 1.0], [1.0, 2.0]])
unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
image = unit_square @ A.T

print(f"A =\n{A}")
print(f"det(A) = {np.linalg.det(A):.6f}")
print(f"单位正方形像的顶点：\n{image}")
print(f"鞋带公式算出的面积 = {polygon_area(image):.6f}")
print(f"|det(A)| = {abs(np.linalg.det(A)):.6f}（应相等）")

# 奇异矩阵：两列共线，面积压成 0
S = np.array([[1.0, 2.0], [2.0, 4.0]])
S_image = unit_square @ S.T
print(f"\n奇异矩阵 S =\n{S}")
print(f"det(S) = {np.linalg.det(S):.3e}，像面积 = {polygon_area(S_image):.3e}")
print(f"rank(S) = {np.linalg.matrix_rank(S)}（降维，面积 0）")


# ---------------------------------------------------------------- 2
section("2. 3×3：行列式 = 体积")
B = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
print(f"B =\n{B}")
print(f"det(B) = {np.linalg.det(B):.6f}")
print(f"单位立方体的像，体积 = |det(B)| = {abs(np.linalg.det(B)):.6f}")

# 用三个列向量的标量三重积验证
c1, c2, c3 = B[:, 0], B[:, 1], B[:, 2]
triple = np.dot(c1, np.cross(c2, c3))
print(f"标量三重积 c1·(c2×c3) = {triple:.6f}（与 det 相同，符号即定向）")

C = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]])
print(f"\n奇异 3×3 矩阵 C 的 det = {np.linalg.det(C):.3e}，rank = {np.linalg.matrix_rank(C)}")


# ---------------------------------------------------------------- 3
section("3. 随机矩阵：det 与网格面积比")
for k in range(3):
    M = rng.uniform(-2.0, 2.0, size=(2, 2))
    d = np.linalg.det(M)
    ratio = grid_area_ratio(M)
    print(f"试验 {k + 1}: det = {d:+.6f}, 网格面积比 = {ratio:.6f}, "
          f"误差 = {abs(ratio - abs(d)):.2e}")
print("结论：无论网格多粗，线性变换把每个小格面积统一乘 |det|，所以总面积比恒为 |det|。")


# ---------------------------------------------------------------- 4
section("4. det(AB) = det(A) det(B)")
print(f"{'n':>4} {'det(A)det(B)':>18} {'det(AB)':>18} {'相对误差':>12}")
for n in (2, 3, 5, 10, 50):
    An = rng.normal(0, 1 / np.sqrt(n), size=(n, n))
    Bn = rng.normal(0, 1 / np.sqrt(n), size=(n, n))
    left = np.linalg.det(An) * np.linalg.det(Bn)
    right = np.linalg.det(An @ Bn)
    rel = abs(right - left) / max(abs(left), 1e-12)
    print(f"{n:>4} {left:>18.6e} {right:>18.6e} {rel:>12.2e}")

An = rng.normal(0, 0.5, size=(10, 10))
while abs(np.linalg.det(An)) < 1e-3:          # 保证可逆，方便验证逆的 det
    An = rng.normal(0, 0.5, size=(10, 10))
print(f"\n10×10 检查：det(Aᵀ) - det(A) = {np.linalg.det(An.T) - np.linalg.det(An):.2e}")
print(f"det(A⁻¹) × det(A) = {np.linalg.det(np.linalg.inv(An)) * np.linalg.det(An):.6f}（应接近 1）")


# ---------------------------------------------------------------- 5
section("5. 奇异与近奇异：det 接近 0")
singular = np.array([[1.0, 2.0], [2.0, 4.0]])
near = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-8]])

for name, M in (("严格奇异", singular), ("近奇异", near)):
    d = np.linalg.det(M)
    cond = np.linalg.cond(M)
    print(f"{name}: det = {d:.3e}, cond = {cond:.3e}, rank = {np.linalg.matrix_rank(M)}")
print("严格奇异：det 数学上为 0，条件数数学上为 ∞；数值上 cond 会给出一个很大的有限值。")
print("近奇异：det 只有 1e-8，条件数却到 1e8；det 的『小』与『病态』方向一致，但刻度依赖尺度。")


# ---------------------------------------------------------------- 6
section("6. det 小 ≠ 病态：det 与条件数无关")
H = np.array([[1.0 / (i + j + 1) for j in range(8)] for i in range(8)])
cases = [
    ("diag(2, 2)", np.diag([2.0, 2.0]), 1.0, "良态，det 正常"),
    ("diag(100, 0.01)", np.diag([100.0, 0.01]), 1e4, "det = 1，但已病态"),
    ("0.1 * I_400", 0.1 * np.eye(400), 1.0, "det 下溢，条件数仍为 1"),
    ("diag(1e6, 1e-6)", np.diag([1e6, 1e-6]), 1e12, "det = 1，条件数 1e12"),
    ("Hilbert 8×8", H, np.linalg.cond(H), "det 极小且病态"),
]

print(f"{'矩阵':<18} {'det':>12} {'log|det|':>12} {'cond':>10}  说明")
for name, M, cond, note in cases:
    d = np.linalg.det(M)
    sign, logabs = np.linalg.slogdet(M)
    print(f"{name:<18} {d:>12.3e} {logabs:>12.3f} {cond:>10.2e}  {note}")

print("\n关键对比：")
print("- 0.1*I_400 的 det 下溢为 0，但它是完美的良态矩阵（cond = 1）。")
print("- diag(1e6,1e-6) 的 det = 1，却病态到 cond = 1e12。")
print("- 所以『det 小』不能推出『病态』；判可逆/病态请用秩、奇异值或条件数。")


# ---------------------------------------------------------------- 7
section("7. Jacobian：极坐标的面积元")
r, theta = 2.0, np.pi / 4
J = np.array([
    [np.cos(theta), -r * np.sin(theta)],
    [np.sin(theta), r * np.cos(theta)],
])
print(f"在 r = {r}, θ = π/4 处，极坐标的 Jacobian =\n{J}")
print(f"det(J) = {np.linalg.det(J):.6f}（应等于 r = {r}）")

dr, dtheta = 0.01, 0.01
polar_cell = np.array([
    [r, theta], [r + dr, theta], [r + dr, theta + dtheta], [r, theta + dtheta],
])
xy_cell = np.column_stack([
    polar_cell[:, 0] * np.cos(polar_cell[:, 1]),
    polar_cell[:, 0] * np.sin(polar_cell[:, 1]),
])
print(f"小扇形实际面积 ≈ {polygon_area(xy_cell):.8f}")
print(f"r·dr·dθ = {r * dr * dtheta:.8f}（面积元公式）")
print("结论：极坐标下 dx dy = r dr dθ，那个 r 就是 Jacobian 行列式。")


# ---------------------------------------------------------------- 8
section("8. 小结")
print("1. |det| 是体积缩放倍率，符号是定向；det = 0 表示降维。")
print("2. 2×2 的 det = 两列叉积；3×3 的 det = 三列标量三重积。")
print("3. det(AB) = det(A)det(B)，所以 det 可乘、det(A⁻¹) = 1/det(A)。")
print("4. 奇异矩阵 det ≈ 0；但 det 的数值大小受尺度与维度影响。")
print("5. 高维用 slogdet 或 LU 对角元求 log|det|，不要直接算 det。")
print("6. 极坐标的 Jacobian 行列式是 r，所以 dx dy = r dr dθ。")

print("\n思考题：")
print("- 把第 3 节的随机矩阵换成 A = [[1, 1], [0, 1]]（剪切），网格面积比是多少？")
print("- 把第 6 节的 0.1*I_400 改成 10*I_400，det 与 cond 分别怎么变？")
print("- 若把第 2 节的单位立方体换成单位球，体积是否仍乘 |det(B)|？为什么？")
