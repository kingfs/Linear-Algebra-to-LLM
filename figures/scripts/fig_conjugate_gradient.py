import numpy as np
import matplotlib.pyplot as plt
import _style as st

A = np.diag([1.0, 9.0])
b = np.array([1.0, 9.0])
x_star = np.array([1.0, 1.0])
x0 = np.array([-1.5, 2.2])


def phi(z):
    return 0.5 * z @ A @ z - b @ z


def gd_path(start, steps=5):
    z = start.copy()
    path = [z.copy()]
    for _ in range(steps):
        r = b - A @ z
        alpha = (r @ r) / (r @ A @ r)
        z = z + alpha * r
        path.append(z.copy())
    return np.array(path)


def cg_path(start, steps=2):
    z = start.copy()
    r = b - A @ z
    p = r.copy()
    path = [z.copy()]
    for _ in range(steps):
        alpha = (r @ r) / (p @ A @ p)
        z = z + alpha * p
        r_new = b - A @ z
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        path.append(z.copy())
    return np.array(path)


gd = gd_path(x0)
cg = cg_path(x0)

fig, ax = plt.subplots(figsize=(7.8, 5.8))
x = np.linspace(-2.1, 1.9, 400)
y = np.linspace(0.1, 2.6, 400)
X, Y = np.meshgrid(x, y)
Z = 0.5 * (X**2 + 9 * Y**2) - X - 9 * Y
levels = [-4.9, -4.7, -4.3, -3.6, -2.5, -1.0]
ax.contour(X, Y, Z, levels=levels, colors="#c9c2b2", linewidths=0.9)

ax.axhline(1.0, color=st.GRAY, lw=0.8, ls=":", alpha=0.6)
ax.axvline(1.0, color=st.GRAY, lw=0.8, ls=":", alpha=0.6)

ax.plot(gd[:, 0], gd[:, 1], "o--", color=st.RED, ms=4.2, lw=1.6, label="gradient descent")
ax.plot(cg[:, 0], cg[:, 1], "o-", color=st.GREEN, ms=5.0, lw=2.2, label="conjugate gradient")
ax.scatter([x0[0]], [x0[1]], color=st.GRAY, s=55, zorder=6)
ax.scatter([x_star[0]], [x_star[1]], color=st.ORANGE, marker="*", s=160, zorder=7)

ax.text(x0[0] - 0.10, x0[1] + 0.10, r"$x_0$", fontsize=11)
ax.text(x_star[0] + 0.06, x_star[1] + 0.10, r"$x^*$", fontsize=12)
ax.text(-2.02, 2.42, r"$A=\mathrm{diag}(1,9),\ b=(1,9)$", fontsize=9)
ax.text(-2.02, 2.20, r"condition number $\kappa=9$", fontsize=9, color=st.GRAY)
ax.text(-0.55, 1.85, "GD zig-zags", fontsize=9, color=st.RED)
ax.text(0.06, 1.43, "CG: 2 steps", fontsize=9, color=st.GREEN)

ax.set_xlim(-2.1, 1.9)
ax.set_ylim(0.1, 2.6)
ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Conjugate directions avoid the zig-zag", loc="left")
st.save(fig, "conjugate-gradient")
