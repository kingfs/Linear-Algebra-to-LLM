import numpy as np
import matplotlib.pyplot as plt
import _style as st

p = np.array([0.9, -0.7])


def f(x, y):
    return 0.5 * (1.8 * x**2 + 0.7 * y**2) + 0.25 * x * y + 0.10 * (x**4 + y**4)


def grad(x, y):
    return np.array([1.8 * x + 0.25 * y + 0.4 * x**3,
                     0.7 * y + 0.25 * x + 0.4 * y**3])


def hess(x, y):
    return np.array([[1.8 + 1.2 * x**2, 0.25],
                     [0.25, 0.7 + 1.2 * y**2]])


g = grad(*p)
H = hess(*p)
w, V = np.linalg.eigh(H)

fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))
ax = axes[0]
x = np.linspace(-1.8, 2.2, 400)
y = np.linspace(-1.9, 1.5, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax.contour(X, Y, Z, levels=14, colors="#b9c2cc", linewidths=0.8)
ax.contour(X, Y, Z, levels=[f(*p)], colors=st.ORANGE, linewidths=1.5)
ax.scatter([p[0]], [p[1]], color=st.RED, s=55, zorder=6)
ax.text(p[0] + 0.08, p[1] + 0.12, r"$p$", color=st.RED, fontsize=12)

for i, color in [(0, st.GREEN), (1, st.ACCENT)]:
    u = V[:, i]
    ax.plot([p[0] - 0.55 * u[0], p[0] + 0.55 * u[0]],
            [p[1] - 0.55 * u[1], p[1] + 0.55 * u[1]],
            color=color, lw=2.0, zorder=5)
    tip = p + 0.62 * u
    ax.text(tip[0] + 0.04, tip[1] + 0.04, rf"$u_{i+1}$", color=color, fontsize=10)

ax.text(-1.70, 1.28, r"$f=\frac{1}{2}(1.8x_1^2+0.7x_2^2)+0.25x_1x_2+0.1(x_1^4+x_2^4)$",
        fontsize=8.5)
ax.text(-1.70, 1.04, rf"$\lambda_1\approx{w[0]:.2f},\ \lambda_2\approx{w[1]:.2f}$",
        fontsize=9, color=st.GRAY)
ax.set_xlim(-1.8, 2.2)
ax.set_ylim(-1.9, 1.5)
ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.grid(True, alpha=0.3)
ax.set_title("Curvature has principal directions", loc="left", fontsize=11)

ax = axes[1]
ts = np.linspace(-1.0, 1.0, 200)
for i, color in [(0, st.GREEN), (1, st.ACCENT)]:
    u = V[:, i]
    exact = np.array([f(*(p + t * u)) - f(*p) for t in ts])
    quad = np.array([g @ (t * u) + 0.5 * (t * u) @ H @ (t * u) for t in ts])
    ax.plot(ts, exact, color=color, lw=2.0, label=rf"$u_{i+1}$ exact")
    ax.plot(ts, quad, color=color, lw=1.5, ls="--", label=rf"$u_{i+1}$ quadratic")

ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 4.0)
ax.set_xlabel(r"step $t$ along $u_i$")
ax.set_ylabel(r"$f(p+tu_i)-f(p)$")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)
ax.set_title("Quadratic model matches to second order", loc="left", fontsize=11)

fig.suptitle("Hessian: local quadratic model of the curvature", y=1.02)
fig.tight_layout()
st.save(fig, "hessian-approximation")
