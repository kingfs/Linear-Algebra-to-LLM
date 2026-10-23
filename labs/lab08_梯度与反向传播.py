"""Lab08 · 梯度与反向传播

对应章节：8.1 梯度 / 8.2 方向导数与 GD / 8.3 Jacobian / 8.4 Hessian / 8.5 矩阵求导

验证：
1. 数值梯度与解析梯度一致（中心差分，误差 ~1e-10 到 1e-6）；
2. f(x) = x^T A x 的梯度是 (A + A^T)x；
3. 二维二次函数上的 GD 轨迹，以及学习率过大时的发散；
4. Hessian 特征值决定曲率与学习率上界 eta < 2 / lambda_max；
5. 两层 ReLU 网络的手写反向传播，与数值梯度逐参数对照。

运行：uv run --with numpy python labs/lab08_梯度与反向传播.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(8)  # 固定种子，保证每次输出一致


def section(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def numeric_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心差分数值梯度，支持任意形状的数组。"""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------- 1
section("1. 数值梯度 vs 解析梯度：f(x, y) = x^2 y + 3 y^2")

def f_poly(v: np.ndarray) -> float:
    x, y = v
    return x * x * y + 3.0 * y * y


point = np.array([1.0, 2.0])
analytic = np.array([2 * point[0] * point[1], point[0] ** 2 + 6 * point[1]])
numeric = numeric_gradient(f_poly, point)
print(f"点 = {point}")
print(f"解析梯度 = {analytic}")
print(f"数值梯度 = {numeric}")
print(f"最大绝对误差 = {max_abs_diff(analytic, numeric):.3e}")
assert max_abs_diff(analytic, numeric) < 1e-6

# ---------------------------------------------------------------- 2
section("2. 二次型：f(x) = x^T A x 的梯度 = (A + A^T)x")

A = rng.normal(size=(3, 3))
x0 = rng.normal(size=3)
print(f"A =\n{A}")
print(f"x = {x0}")
print(f"A 对称吗？{np.allclose(A, A.T)}")


def f_quad(v: np.ndarray) -> float:
    return float(v @ A @ v)


analytic_q = (A + A.T) @ x0
numeric_q = numeric_gradient(f_quad, x0.copy())
print(f"(A + A^T)x = {analytic_q}")
print(f"数值梯度     = {numeric_q}")
print(f"最大绝对误差 = {max_abs_diff(analytic_q, numeric_q):.3e}")
assert max_abs_diff(analytic_q, numeric_q) < 1e-6

# 对称化后再验证一次：此时梯度就是 2Ax
S = (A + A.T) / 2
analytic_s = 2 * S @ x0
numeric_s = numeric_gradient(lambda v: float(v @ S @ v), x0.copy())
print(f"对称部分 S 的 2Sx 与数值梯度误差 = {max_abs_diff(analytic_s, numeric_s):.3e}")

# ---------------------------------------------------------------- 3
section("3. 二维 GD 轨迹：f(x) = 0.5 * (3 x1^2 + x2^2)")

A2 = np.diag([3.0, 1.0])


def f2d(v: np.ndarray) -> float:
    return 0.5 * float(v @ A2 @ v)


def grad2d(v: np.ndarray) -> np.ndarray:
    return A2 @ v


def run_gd(x_start: np.ndarray, eta: float, steps: int) -> list[np.ndarray]:
    path = [x_start.copy()]
    x = x_start.copy()
    for _ in range(steps):
        x = x - eta * grad2d(x)
        path.append(x.copy())
    return path


for eta in (0.4, 0.7):
    path = run_gd(np.array([1.0, 1.0]), eta, 5)
    print(f"\n学习率 eta = {eta}（上界 2/lambda_max = {2 / 3:.3f}）")
    print("  步      x1        x2        f(x)")
    for k, xk in enumerate(path):
        print(f"  {k:>2}  {xk[0]:>9.5f} {xk[1]:>9.5f} {f2d(xk):>9.5f}")
    if eta < 2 / 3:
        assert all(f2d(path[i + 1]) < f2d(path[i]) for i in range(len(path) - 1))
    else:
        assert f2d(path[-1]) > f2d(path[0])

# ---------------------------------------------------------------- 4
section("4. Hessian 特征值 → 曲率与学习率上界")

eigs = np.linalg.eigvalsh(A2)
eta_bound = 2 / eigs.max()
print(f"Hessian = diag(3, 1) 的特征值 = {eigs}")
print(f"理论收敛上界：eta < 2 / {eigs.max():.1f} = {eta_bound:.4f}")
print("\n   eta     max|1-eta*lambda|   30 步后 ||x||      判断")
x_start = np.array([1.0, 1.0])
for eta in (0.05, 0.3, 0.6, 0.7):
    path = run_gd(x_start, eta, 30)
    rho = float(np.max(np.abs(1 - eta * eigs)))
    final_norm = float(np.linalg.norm(path[-1]))
    verdict = "收敛" if rho < 1 else "发散"
    print(f"  {eta:>5.2f}   {rho:>16.4f}   {final_norm:>16.3e}      {verdict}")
    assert (rho < 1) == (final_norm < np.linalg.norm(x_start))

print("\n条件数 kappa = 3/1 = 3；kappa 越大，慢方向收缩越慢（见 8.7 节）。")

# ---------------------------------------------------------------- 5
section("5. 两层 ReLU 网络：手写反向传播 vs 数值梯度")

# 固定一个小网络：2 -> 2 (ReLU) -> 1
x_in = np.array([1.0, -1.0])
W1 = np.array([[2.0, 1.0], [1.0, 1.0]])
b1 = np.array([0.5, -0.5])
w2 = np.array([1.0, 1.0])
y_true = 1.0


def forward(params: dict, x: np.ndarray) -> tuple[float, tuple]:
    z1 = params["W1"] @ x + params["b1"]
    h = np.maximum(z1, 0.0)
    out = params["w2"] @ h
    return out, (x, z1, h)


def loss_fn(params: dict) -> float:
    out, _ = forward(params, x_in)
    return 0.5 * (out - y_true) ** 2


def backward(params: dict) -> dict:
    out, (x, z1, h) = forward(params, x_in)
    d_out = out - y_true            # dL/do
    d_w2 = d_out * h                 # dL/dw2
    d_z1 = d_out * params["w2"] * (z1 > 0)  # 穿过 ReLU
    d_W1 = np.outer(d_z1, x)         # dL/dW1 = delta * x^T
    d_b1 = d_z1
    return {"W1": d_W1, "b1": d_b1, "w2": d_w2}


params = {"W1": W1, "b1": b1, "w2": w2}
out, cache = forward(params, x_in)
print(f"前向：z1 = {cache[1]}，h = {cache[2]}，输出 = {out}，loss = {loss_fn(params):.6f}")

grads = backward(params)
print("\n解析梯度：")
for name in ("W1", "b1", "w2"):
    print(f"  dL/d{name} =\n{grads[name]}" if grads[name].ndim > 1
          else f"  dL/d{name} = {grads[name]}")

print("\n逐参数数值梯度对照：")
for name in ("W1", "b1", "w2"):
    def f_param(p, name=name):
        return loss_fn({**params, name: p})

    num = numeric_gradient(f_param, params[name].copy())
    err = max_abs_diff(grads[name], num)
    print(f"  {name}: 最大绝对误差 = {err:.3e}")
    assert err < 1e-6

# 用解析梯度做几步，确认 loss 真的下降
lr = 0.2
print(f"\n用解析梯度做 10 步（学习率 {lr}）：")
for step in range(11):
    if step % 2 == 0:
        print(f"  step {step:>2}: loss = {loss_fn(params):.8f}")
    g = backward(params)
    for name in params:
        params[name] = params[name] - lr * g[name]

print("\n结论：手写反向传播的梯度与数值梯度一致；形状约定（梯度与参数同形）")
print("      和链式法则（上游梯度 × 本地 Jacobian）是整套流程的两根支柱。")

# ---------------------------------------------------------------- 思考题
# 1. 把第 5 节的 ReLU 换成 sigmoid，反向传播要改哪一行？
# 2. 第 4 节里 eta 恰好等于 2/lambda_max 时，哪个方向会出问题？跑一跑验证。
# 3. 若 f(x) = x^T A x 且 A 不对称，为什么梯度不是 2Ax？
