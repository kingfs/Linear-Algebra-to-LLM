import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GRAY, save

y = np.array([2.15, 2.55])
p = np.array([2.15, 0.0])
r = y - p

fig, ax = plt.subplots(figsize=(5.4, 3.9))
ax.set_aspect("equal")

# 列空间（侧视图里画成一条直线）
ax.plot([-0.6, 5.3], [0, 0], color=ACCENT, lw=2.4)
ax.text(4.55, -0.42, r"$\mathrm{col}(A)$", color=ACCENT, fontsize=13)
ax.text(5.05, -0.05, "plane", color=GRAY, fontsize=9, style="italic")

ax.annotate("", xy=y, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.4))
ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4))
ax.plot([p[0], y[0]], [p[1], y[1]], "--", color=GRAY, lw=1.5)

# 直角标记
t = 0.22
ax.plot([p[0] - t, p[0] - t, p[0]], [0, t, t], color=GRAY, lw=1.0)

ax.text(y[0] + 0.14, y[1] - 0.1, r"$\mathbf{y}$", color=GRAY, fontsize=15)
ax.text(p[0] + 0.14, p[1] + 0.1, r"$\hat{\mathbf{y}}=A\mathbf{x}$", color=ORANGE, fontsize=14)
ax.text(p[0] + 0.16, 1.15, r"$\mathbf{r}=\mathbf{y}-\hat{\mathbf{y}}$", color=GRAY, fontsize=13)

ax.set_xlim(-0.7, 5.4)
ax.set_ylim(-0.7, 3.0)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Least squares: drop a perpendicular onto col(A)", loc="left", fontsize=10.5)
save(fig, "least-squares")
