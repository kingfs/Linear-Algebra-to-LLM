import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
from _style import ACCENT, GREEN, ORANGE, GRAY, save

a = np.array([2.7, 0.35])
b = np.array([0.95, 1.85])
c = np.array([0.0, 2.35])

fig, ax = plt.subplots(figsize=(4.9, 4.2))
ax.set_aspect("equal")

plane = Polygon([(0, 0), a, a + b, b], closed=True, facecolor=ACCENT,
                alpha=0.10, edgecolor=ACCENT, lw=1.2)
ax.add_patch(plane)

ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.4))
ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4))
ax.annotate("", xy=c, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.8))

ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=8, theta2=63, color=GRAY, lw=1.0))

ax.text(a[0] + 0.08, a[1] - 0.22, r"$\mathbf{a}$", color=ACCENT, fontsize=14)
ax.text(b[0] - 0.30, b[1] + 0.08, r"$\mathbf{b}$", color=GREEN, fontsize=14)
ax.text(c[0] + 0.12, c[1] - 0.18, r"$\mathbf{a}\times\mathbf{b}$", color=ORANGE, fontsize=14)
ax.text(0.62, 1.55, "normal", color=GRAY, fontsize=10, style="italic")

ax.set_xlim(-0.45, 3.9)
ax.set_ylim(-0.5, 2.9)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Cross product: normal to the plane of a and b", loc="left", fontsize=11)
save(fig, "cross-product")
