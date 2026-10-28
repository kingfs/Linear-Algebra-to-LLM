import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d

fig, ax = plt.subplots(figsize=(6.0, 5.6))
axes2d(ax)

theta = np.linspace(0, 2 * np.pi, 400)
circle = np.c_[np.cos(theta), np.sin(theta)]
ax.fill(circle[:, 0], circle[:, 1], color=ACCENT, alpha=0.08)
ax.plot(circle[:, 0], circle[:, 1], color=ACCENT, lw=2.0)

diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
ax.fill(diamond[:, 0], diamond[:, 1], color=ORANGE, alpha=0.08)
ax.plot(diamond[:, 0], diamond[:, 1], color=ORANGE, lw=2.0)

square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
ax.fill(square[:, 0], square[:, 1], color=GREEN, alpha=0.06)
ax.plot(square[:, 0], square[:, 1], color=GREEN, lw=2.0)

# 同一方向与三个单位球的交点
d = np.array([0.6, 0.8])
points = {
    "L2": d / np.linalg.norm(d),
    "L1": d / (abs(d[0]) + abs(d[1])),
    "Linf": d / max(abs(d[0]), abs(d[1])),
}
ax.plot([0, 1.55], [0, 2.05], color=GRAY, lw=1.1, ls="--", alpha=0.9)
for name, color in [("L2", ACCENT), ("L1", ORANGE), ("Linf", GREEN)]:
    p = points[name]
    ax.plot(*p, "o", color=color, ms=7, mec="white", mew=1.2, zorder=5)

ax.text(0.58, 0.64, r"$L_2$", color=ACCENT, fontsize=13)
ax.text(0.30, -1.28, r"$L_1$", color=ORANGE, fontsize=13)
ax.text(1.02, 1.02, r"$L_\infty$", color=GREEN, fontsize=13)
ax.text(0.98, 1.52, "same direction", color=GRAY, fontsize=9, rotation=53)

ax.set_xlim(-1.75, 1.85)
ax.set_ylim(-1.65, 2.3)
ax.set_title("Unit balls: three definitions of length 1", loc="left")
save(fig, "norm-balls")
