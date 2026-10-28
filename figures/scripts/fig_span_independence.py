import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d, arrow

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
for ax in axes:
    axes2d(ax)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)

# ---------------------------------------------------------------- 左：相关
ax = axes[0]
v1 = np.array([1.0, 0.5])
v2 = 1.5 * v1
c1, c2 = np.meshgrid(np.linspace(-0.9, 0.9, 19), np.linspace(-0.9, 0.9, 19))
pts = c1[..., None] * v1 + c2[..., None] * v2
ax.scatter(pts[..., 0], pts[..., 1], s=7, color=ORANGE, alpha=0.35, linewidths=0)

line = np.array([-2.3, 2.3])[:, None] * v1
ax.plot(line[:, 0], line[:, 1], color=ORANGE, lw=2.0, alpha=0.55)
arrow(ax, (0, 0), v1, ACCENT)
arrow(ax, (0, 0), v2, ORANGE)
ax.text(0.86, 0.30, r"$\mathbf{v}_1$", color=ACCENT, fontsize=13)
ax.text(1.28, 0.58, r"$\mathbf{v}_2=1.5\mathbf{v}_1$", color=ORANGE, fontsize=11)
ax.text(-2.25, 1.72, "span = a line (1D)", color=ORANGE, fontsize=11)
ax.set_title("Dependent: span is a line", loc="left")

# ---------------------------------------------------------------- 右：无关
ax = axes[1]
v1 = np.array([1.0, 0.15])
v2 = np.array([0.25, 1.0])
c1, c2 = np.meshgrid(np.linspace(-1.5, 1.5, 19), np.linspace(-1.5, 1.5, 19))
pts = c1[..., None] * v1 + c2[..., None] * v2
ax.scatter(pts[..., 0], pts[..., 1], s=7, color=ACCENT, alpha=0.30, linewidths=0)

para = np.array([[0, 0], v1, v1 + v2, v2])
ax.add_patch(Polygon(para, closed=True, facecolor=GREEN, alpha=0.12, edgecolor=GREEN, lw=1.4))
arrow(ax, (0, 0), v1, ACCENT)
arrow(ax, (0, 0), v2, ORANGE)
ax.text(0.82, -0.08, r"$\mathbf{v}_1$", color=ACCENT, fontsize=13)
ax.text(0.30, 0.86, r"$\mathbf{v}_2$", color=ORANGE, fontsize=13)
ax.text(-2.25, 1.72, "span = the whole plane (2D)", color=ACCENT, fontsize=11)
ax.set_title("Independent: span is a plane", loc="left")

fig.subplots_adjust(wspace=0.18)
save(fig, "span-independence")
