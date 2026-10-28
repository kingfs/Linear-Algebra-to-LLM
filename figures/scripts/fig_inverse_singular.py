import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d

A = np.array([[1.6, 0.7], [0.2, 1.3]])
S = np.array([[1.0, 1.0], [1.0, 1.0]])
unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def draw_grid(ax, M, color, alpha, lw, extent=1.8, n=9):
    for t in np.linspace(-extent, extent, n):
        seg = np.array([[-extent, t], [extent, t]]) @ M.T
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, alpha=alpha, zorder=1)
        seg = np.array([[t, -extent], [t, extent]]) @ M.T
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, alpha=alpha, zorder=1)


fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8))

# ---------------------------------------------------------------- 左：可逆
ax = axes[0]
axes2d(ax, grid=False)
draw_grid(ax, np.eye(2), GRAY, 0.45, 0.7)
draw_grid(ax, A, ACCENT, 0.30, 0.9)
poly = unit @ A.T
ax.add_patch(Polygon(poly, closed=True, facecolor=ACCENT, alpha=0.16, edgecolor=ACCENT, lw=1.8, zorder=3))
ax.plot(*np.vstack([unit, unit[:1]]).T, color=GRAY, lw=1.1, ls="--", zorder=2)
ax.text(1.55, 1.63, "det A = 1.94 > 0", color=ACCENT, fontsize=10)
ax.text(-0.98, 1.42, r"$A^{-1}$ exists", color=ORANGE, fontsize=11)
ax.text(-0.98, 1.12, "area is preserved", color=GRAY, fontsize=10)
ax.set_xlim(-1.1, 2.9)
ax.set_ylim(-0.8, 2.7)
ax.set_title("Invertible: still a 2D parallelogram", loc="left")

# ---------------------------------------------------------------- 右：奇异
ax = axes[1]
axes2d(ax, grid=False)
draw_grid(ax, np.eye(2), GRAY, 0.45, 0.7)
draw_grid(ax, S, ORANGE, 0.30, 0.9)
ax.plot([0, 2], [0, 2], color=ORANGE, lw=3.0, alpha=0.9, zorder=3)
ax.text(1.32, 1.48, "det S = 0", color=ORANGE, fontsize=10)
ax.text(-2.35, 1.72, "collapses to a line", color=ORANGE, fontsize=11)
ax.text(-2.35, 1.44, "different inputs share one output", color=GRAY, fontsize=9)

# 两个不同输入映到同一点
u = np.array([0.4, 0.4])
v = np.array([1.4, -0.6])
out = S @ u
for point, color, label, offset in [
    (u, ACCENT, r"$\mathbf{x}$", (-0.30, 0.16)),
    (v, GREEN, r"$\mathbf{x}+\mathbf{n}$", (0.10, -0.32)),
    (out, ORANGE, "same output", (0.12, 0.14)),
]:
    ax.plot(*point, "o", color=color, ms=7, mec="white", mew=1.0, zorder=5)
    ax.text(point[0] + offset[0], point[1] + offset[1], label, color=color, fontsize=10)
ax.plot([u[0], out[0]], [u[1], out[1]], ":", color=ACCENT, lw=1.2, zorder=2)
ax.plot([v[0], out[0]], [v[1], out[1]], ":", color=GREEN, lw=1.2, zorder=2)
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-1.5, 2.6)
ax.set_title("Singular: the plane collapses to a line", loc="left")

fig.subplots_adjust(wspace=0.18)
save(fig, "inverse-singular")
