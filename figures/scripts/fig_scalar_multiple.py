import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, GREEN, ORANGE, GRAY, save, axes2d, arrow

v = np.array([1.5, 0.75])

fig, ax = plt.subplots(figsize=(6.2, 3.9))
axes2d(ax)

# 整条直线：数乘只能在它上面移动
s = np.linspace(-1.75, 2.35, 2)
line = s[:, None] * v
ax.plot(line[:, 0], line[:, 1], "--", color=GRAY, lw=1.3, alpha=0.85, zorder=1)

arrow(ax, (0, 0), v, ACCENT, lw=2.4)
arrow(ax, (0, 0), 1.8 * v, GREEN, lw=2.4)
arrow(ax, (0, 0), -1.3 * v, ORANGE, lw=2.4)
arrow(ax, (0, 0), 0.5 * v, GRAY, lw=1.8)

ax.text(v[0] + 0.08, v[1] + 0.10, r"$\mathbf{v}$", color=ACCENT, fontsize=14)
ax.text(1.8 * v[0] - 0.28, 1.8 * v[1] + 0.12, r"$1.8\mathbf{v}$", color=GREEN, fontsize=13)
ax.text(-1.3 * v[0] - 0.62, -1.3 * v[1] - 0.24, r"$-1.3\mathbf{v}$", color=ORANGE, fontsize=13)
ax.text(0.5 * v[0] + 0.08, 0.5 * v[1] - 0.24, r"$0.5\mathbf{v}$", color=GRAY, fontsize=12)
ax.text(-0.12, -0.24, r"$c=0$", color=GRAY, fontsize=10)

ax.text(-2.55, 1.85,
        r"$c>1$ stretch  |  $0<c<1$ shrink  |  $c<0$ flip",
        fontsize=9.5, color=GRAY)
ax.text(-2.55, -1.55,
        "every $c\\mathbf{v}$ stays on the same line through the origin",
        fontsize=10, color=ACCENT)

ax.set_xlim(-2.8, 3.4)
ax.set_ylim(-1.7, 2.1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Scalar multiplication: length and sign change, line does not",
             loc="left", fontsize=11)
save(fig, "scalar-multiple")
