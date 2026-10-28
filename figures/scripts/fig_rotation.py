import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d, arrow

R = np.array([[0.0, -1.0], [1.0, 0.0]])
x = np.array([1.0, 0.0])
y = np.array([1.0, 1.0])

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5))
for ax in axes:
    axes2d(ax)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.05, 2.05)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])

# ---------------------------------------------------------------- 左：一次 R
ax = axes[0]
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
ax.plot(square[:, 0], square[:, 1], color=GRAY, lw=1.1, ls="--", alpha=0.9)

arrow(ax, (0, 0), x, ACCENT)
arrow(ax, (0, 0), y, GREEN)
arrow(ax, (0, 0), R @ x, ORANGE)
arrow(ax, (0, 0), R @ y, ORANGE)

ax.text(0.78, -0.30, r"$\mathbf{x}=(1,0)$", color=ACCENT, fontsize=11)
ax.text(0.92, 1.10, r"$\mathbf{y}=(1,1)$", color=GREEN, fontsize=11)
ax.text(-0.52, 1.02, r"$R\mathbf{x}=(0,1)$", color=ORANGE, fontsize=11)
ax.text(-1.78, 0.90, r"$R\mathbf{y}=(-1,1)$", color=ORANGE, fontsize=11)

ax.add_patch(Arc((0, 0), 1.0, 1.0, theta1=0, theta2=90, color=ORANGE, lw=1.3, ls="--"))
ax.text(0.60, 0.66, r"$90^\circ$", color=ORANGE, fontsize=11)
ax.set_title("One R: rotate every vector by 90°", loc="left")

# ---------------------------------------------------------------- 右：两次 R
ax = axes[1]
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.0, ls=":", alpha=0.8)

Rx = R @ x
R2x = R @ Rx
arrow(ax, (0, 0), x, ACCENT)
arrow(ax, (0, 0), Rx, ORANGE)
arrow(ax, (0, 0), R2x, GREEN)

ax.text(0.70, -0.32, r"$\mathbf{x}$", color=ACCENT, fontsize=13)
ax.text(-0.44, 0.92, r"$R\mathbf{x}$", color=ORANGE, fontsize=13)
ax.text(-0.94, -0.44, r"$R^2\mathbf{x}$", color=GREEN, fontsize=13)

ax.add_patch(Arc((0, 0), 1.35, 1.35, theta1=0, theta2=90, color=ORANGE, lw=1.3, ls="--"))
ax.text(0.28, 0.98, r"$90^\circ$", color=ORANGE, fontsize=10)
ax.add_patch(Arc((0, 0), 1.75, 1.75, theta1=90, theta2=180, color=GREEN, lw=1.3, ls="--"))
ax.text(-1.18, 0.48, r"$90^\circ$", color=GREEN, fontsize=10)
ax.text(-0.55, -1.55, r"$R^2=-I$", color=GREEN, fontsize=12)
ax.set_title("Two R's: 180° rotation, $R^2=-I$", loc="left")

fig.subplots_adjust(wspace=0.18)
save(fig, "rotation")
