import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d, arrow

x = np.array([1.0, 0.0])
y = np.array([0.6, 0.8])
p = np.array([0.6, 0.0])
theta = np.degrees(np.arccos(0.6))

fig, ax = plt.subplots(figsize=(5.8, 5.2))
axes2d(ax)

t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), color=GRAY, lw=1.0, ls=":", alpha=0.8)

arrow(ax, (0, 0), x, ACCENT)
arrow(ax, (0, 0), y, GREEN)
ax.plot([y[0], p[0]], [y[1], p[1]], "--", color=GRAY, lw=1.4)
ax.plot([p[0], p[0]], [0, y[1]], "--", color=ORANGE, lw=1.2, alpha=0.7)
ax.plot([0, p[0]], [0, 0], color=ORANGE, lw=3.0, alpha=0.45)

# 直角标记
ax.plot([0.55, 0.55, 0.60], [0, 0.05, 0.05], color=GRAY, lw=1.0)
ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=theta, color=ORANGE, lw=1.4))

ax.text(0.80, -0.26, r"$\mathbf{x}$", color=ACCENT, fontsize=14)
ax.text(0.64, 0.86, r"$\mathbf{y}$", color=GREEN, fontsize=14)
ax.text(0.44, 0.19, r"$\theta$", color=ORANGE, fontsize=12)
ax.text(0.66, 0.33, r"$\cos\theta=0.6$", color=ORANGE, fontsize=12)
ax.text(0.60, -0.19, r"$\mathbf{p}$", color=ORANGE, fontsize=12)
ax.text(0.74, 0.70, "unit circle", color=GRAY, fontsize=9, rotation=-38)

ax.set_xlim(-1.35, 1.45)
ax.set_ylim(-1.25, 1.35)
ax.set_title("Cosine = shadow on the unit direction", loc="left")
save(fig, "cosine-angle")
