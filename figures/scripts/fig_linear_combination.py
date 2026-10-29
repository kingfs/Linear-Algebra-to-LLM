import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, GREEN, ORANGE, GRAY, save, axes2d, arrow

v1 = np.array([2.0, 0.5])
v2 = np.array([0.4, 1.8])

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))

# ---------------------------------------------------------------- 左：正系数
ax = axes[0]
axes2d(ax)

c1, c2 = 0.9, 0.8
w = c1 * v1 + c2 * v2
p1, p2 = c1 * v1, c2 * v2

arrow(ax, (0, 0), v1, ACCENT, lw=2.0)
arrow(ax, (0, 0), v2, GREEN, lw=2.0)
arrow(ax, (0, 0), p1, ACCENT, lw=1.5, ls="--")
arrow(ax, (0, 0), p2, GREEN, lw=1.5, ls="--")
arrow(ax, (0, 0), w, ORANGE, lw=2.6)

ax.plot([p1[0], w[0]], [p1[1], w[1]], "--", color=GRAY, lw=1.0)
ax.plot([p2[0], w[0]], [p2[1], w[1]], "--", color=GRAY, lw=1.0)

ax.text(v1[0] + 0.05, v1[1] + 0.10, r"$\mathbf{v}_1$", color=ACCENT, fontsize=13)
ax.text(v2[0] + 0.10, v2[1] + 0.02, r"$\mathbf{v}_2$", color=GREEN, fontsize=13)
ax.text(w[0] + 0.08, w[1] + 0.08, r"$\mathbf{w}$", color=ORANGE, fontsize=14)
ax.text(p1[0] - 0.50, p1[1] - 0.30, r"$0.9\mathbf{v}_1$", color=ACCENT, fontsize=9)
ax.text(p2[0] - 0.62, p2[1] + 0.10, r"$0.8\mathbf{v}_2$", color=GREEN, fontsize=9)

ax.text(-1.35, 2.14, r"$\mathbf{w}=0.9\mathbf{v}_1+0.8\mathbf{v}_2$", fontsize=11)
ax.text(-1.35, -1.35, "positive coefficients: inside the cone", fontsize=9.5, color=GRAY)
ax.set_title("Positive recipe", loc="left", fontsize=11)

# ---------------------------------------------------------------- 右：负系数
ax = axes[1]
axes2d(ax)

c1, c2 = 1.5, -0.9
w = c1 * v1 + c2 * v2
p1, p2 = c1 * v1, c2 * v2

arrow(ax, (0, 0), v1, ACCENT, lw=2.0)
arrow(ax, (0, 0), v2, GREEN, lw=2.0)
arrow(ax, (0, 0), p1, ACCENT, lw=1.5, ls="--")
arrow(ax, (0, 0), p2, GREEN, lw=1.5, ls="--")
arrow(ax, (0, 0), w, ORANGE, lw=2.6)

ax.plot([p1[0], w[0]], [p1[1], w[1]], "--", color=GRAY, lw=1.0)
ax.plot([p2[0], w[0]], [p2[1], w[1]], "--", color=GRAY, lw=1.0)

ax.text(v1[0] + 0.05, v1[1] + 0.10, r"$\mathbf{v}_1$", color=ACCENT, fontsize=13)
ax.text(v2[0] + 0.10, v2[1] + 0.02, r"$\mathbf{v}_2$", color=GREEN, fontsize=13)
ax.text(w[0] + 0.08, w[1] - 0.18, r"$\mathbf{w}$", color=ORANGE, fontsize=14)
ax.text(p1[0] - 0.55, p1[1] + 0.12, r"$1.5\mathbf{v}_1$", color=ACCENT, fontsize=9)
ax.text(p2[0] + 0.08, p2[1] + 0.10, r"$-0.9\mathbf{v}_2$", color=GREEN, fontsize=9)

ax.text(-1.35, 2.14, r"$\mathbf{w}=1.5\mathbf{v}_1-0.9\mathbf{v}_2$", fontsize=11)
ax.set_title("A negative coefficient is still legal", loc="left", fontsize=11)

for ax in axes:
    ax.set_xlim(-1.5, 3.6)
    ax.set_ylim(-1.85, 2.35)
    ax.set_xticks([])
    ax.set_yticks([])

fig.subplots_adjust(wspace=0.15)
save(fig, "linear-combination")
