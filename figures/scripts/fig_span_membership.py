import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _style import ACCENT, GREEN, ORANGE, GRAY, save, arrow

E1 = np.array([1.5, 0.0, 0.0])
E2 = np.array([0.0, 1.5, 0.0])


def project(p):
    """Oblique 3D -> 2D: z goes up, x/y span the floor plane."""
    x, y, z = p
    return np.array([x - 0.62 * y, 0.34 * x + 0.34 * y + 0.92 * z])


def draw_plane(ax):
    u = np.linspace(-1.0, 1.0, 9)
    for k in u:
        pts = np.array([project(k * E1 + t * E2) for t in np.linspace(-1, 1, 40)])
        ax.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=0.55, alpha=0.35, zorder=1)
        pts = np.array([project(t * E1 + k * E2) for t in np.linspace(-1, 1, 40)])
        ax.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=0.55, alpha=0.35, zorder=1)

    corners = np.array([
        project(-E1 - E2), project(E1 - E2), project(E1 + E2), project(-E1 + E2)
    ])
    ax.add_patch(Polygon(corners, closed=True, facecolor=ACCENT, alpha=0.07,
                         edgecolor=ACCENT, lw=1.0, zorder=0))


fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.0))

# ---------------------------------------------------------------- 左：在张成内
ax = axes[0]
draw_plane(ax)

b_in = np.array([0.7, 0.9, 0.0])
arrow(ax, (0, 0), project(E1), ACCENT, lw=2.0, zorder=5)
arrow(ax, (0, 0), project(E2), GREEN, lw=2.0, zorder=5)
arrow(ax, (0, 0), project(b_in), ORANGE, lw=2.6, zorder=6)

ax.text(*project(E1) + np.array([0.06, -0.14]), r"$\mathbf{v}_1$",
        color=ACCENT, fontsize=12)
ax.text(*project(E2) + np.array([-0.22, 0.06]), r"$\mathbf{v}_2$",
        color=GREEN, fontsize=12)
ax.text(*project(b_in) + np.array([0.08, 0.04]), r"$\mathbf{b}$",
        color=ORANGE, fontsize=14)
ax.text(-2.62, -1.16, r"$\mathrm{span}\{\mathbf{v}_1,\mathbf{v}_2\}$",
        color=ACCENT, fontsize=11)
ax.text(-2.62, 1.62, r"$\mathbf{b}\in\mathrm{span}$", color=ORANGE, fontsize=11)
ax.set_title("Inside the span: an exact recipe exists", loc="left", fontsize=10.5)

# ---------------------------------------------------------------- 右：在张成外
ax = axes[1]
draw_plane(ax)

b_out = np.array([0.45, 0.50, 1.45])
p = np.array([0.45, 0.50, 0.0])

# z 轴：提示「平面外还有方向」
ax.plot([0, 0], [0, project(np.array([0, 0, 1.7]))[1]],
        ":", color=GRAY, lw=1.1, zorder=2)
ax.text(-0.08, project(np.array([0, 0, 1.20]))[1],
        "extra direction", color=GRAY, fontsize=8.5, ha="right")

arrow(ax, (0, 0), project(E1), ACCENT, lw=2.0, zorder=5)
arrow(ax, (0, 0), project(E2), GREEN, lw=2.0, zorder=5)
arrow(ax, (0, 0), project(b_out), GRAY, lw=2.4, zorder=6)
arrow(ax, (0, 0), project(p), ORANGE, lw=2.4, zorder=6)
ax.plot([project(p)[0], project(b_out)[0]], [project(p)[1], project(b_out)[1]],
        "--", color=ORANGE, lw=1.4, zorder=7)

ax.text(*project(E1) + np.array([0.06, -0.14]), r"$\mathbf{v}_1$",
        color=ACCENT, fontsize=12)
ax.text(*project(E2) + np.array([-0.22, 0.06]), r"$\mathbf{v}_2$",
        color=GREEN, fontsize=12)
ax.text(*project(b_out) + np.array([0.08, 0.02]), r"$\mathbf{b}$",
        color=GRAY, fontsize=14)
ax.text(*project(p) + np.array([0.08, -0.20]), r"$\mathbf{p}$",
        color=ORANGE, fontsize=14)
ax.text(*project(p) + np.array([0.20, 0.34]), r"residual", color=ORANGE, fontsize=9)

ax.text(-2.62, -1.16, r"$\mathrm{span}\{\mathbf{v}_1,\mathbf{v}_2\}$",
        color=ACCENT, fontsize=11)
ax.text(-2.62, 1.62, r"$\mathbf{b}\notin\mathrm{span}$", color=GRAY, fontsize=11)
ax.set_title("Outside the span: only the nearest point", loc="left", fontsize=10.5)

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-1.35, 2.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

fig.subplots_adjust(wspace=0.05)
save(fig, "span-membership")
