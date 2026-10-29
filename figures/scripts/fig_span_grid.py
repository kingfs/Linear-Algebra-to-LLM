import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, GREEN, ORANGE, GRAY, save, axes2d, arrow

v1 = np.array([1.5, 0.3])
v2 = np.array([0.4, 1.1])
c_star = np.array([1.2, 0.8])
w = c_star[0] * v1 + c_star[1] * v2

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.7),
                         gridspec_kw={"width_ratios": [1.0, 1.25]})

# ---------------------------------------------------------------- 左：系数平面
ax = axes[0]
axes2d(ax)

lo, hi = -1.6, 1.6
for k in np.arange(-1.5, 1.51, 0.5):
    ax.axhline(k, color=GRAY, lw=0.6, alpha=0.45, zorder=1)
    ax.axvline(k, color=GRAY, lw=0.6, alpha=0.45, zorder=1)

ax.scatter([c_star[0]], [c_star[1]], s=46, color=ORANGE, zorder=5)
ax.plot([c_star[0], c_star[0]], [0, c_star[1]], "--", color=ORANGE, lw=1.0)
ax.plot([0, c_star[0]], [c_star[1], c_star[1]], "--", color=ORANGE, lw=1.0)
ax.annotate(r"$(c_1,c_2)=(1.2,0.8)$", xy=(c_star[0], c_star[1]),
            xytext=(-1.55, 1.38), color=ORANGE, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xticks([-1.5, -0.5, 0.5, 1.5])
ax.set_yticks([-1.5, -0.5, 0.5, 1.5])
ax.set_xlabel(r"$c_1$")
ax.set_ylabel(r"$c_2$")
ax.set_title("Coefficient space: pick a pair", loc="left", fontsize=11)

# ---------------------------------------------------------------- 右：向量平面
ax = axes[1]
axes2d(ax)

# 把系数平面的网格线逐条映射过去，得到的是一张斜网格
cs = np.linspace(-1.5, 1.5, 121)
for k in np.arange(-1.5, 1.51, 0.5):
    pts = k * v1 + cs[:, None] * v2
    ax.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=0.55, alpha=0.38, zorder=1)
    pts = cs[:, None] * v1 + k * v2
    ax.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=0.55, alpha=0.38, zorder=1)

# 窗口边界
corners = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]])
box = corners[:, :1] * v1 + corners[:, 1:] * v2
ax.plot(box[:, 0], box[:, 1], color=GRAY, lw=1.1, ls="--", alpha=0.8, zorder=2)

arrow(ax, (0, 0), v1, ACCENT, lw=2.2, zorder=6)
arrow(ax, (0, 0), v2, GREEN, lw=2.2, zorder=6)
arrow(ax, (0, 0), w, ORANGE, lw=2.6, zorder=7)

ax.scatter([w[0]], [w[1]], s=46, color=ORANGE, zorder=8)
ax.text(v1[0] + 0.06, v1[1] - 0.16, r"$\mathbf{v}_1$", color=ACCENT, fontsize=13)
ax.text(v2[0] + 0.08, v2[1] + 0.04, r"$\mathbf{v}_2$", color=GREEN, fontsize=13)
ax.text(w[0] + 0.08, w[1] + 0.06, r"$\mathbf{w}$", color=ORANGE, fontsize=14)

ax.text(-3.05, 2.02,
        r"$\mathbf{w}=1.2\mathbf{v}_1+0.8\mathbf{v}_2$", fontsize=10.5)
ax.text(-3.05, -2.28,
        "the dashed box is only a window; all real $c_i$ fill the plane",
        fontsize=8.8, color=GRAY)

ax.set_xlim(-3.25, 3.25)
ax.set_ylim(-2.45, 2.35)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r"Vector space: the image is $\mathrm{span}\{\mathbf{v}_1,\mathbf{v}_2\}$",
             loc="left", fontsize=11)

fig.subplots_adjust(wspace=0.16)
save(fig, "span-grid")
