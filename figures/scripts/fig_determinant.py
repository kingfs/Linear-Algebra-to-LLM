import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _style import ACCENT, ORANGE, save

A = np.array([[2.0, 1.0], [0.0, 1.5]])
unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
trans = unit @ A.T

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.set_aspect("equal")
ax.grid(True, alpha=0.7)
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.add_patch(Polygon(unit, closed=True, facecolor=ACCENT, alpha=0.10, edgecolor=ACCENT, lw=1.8))
ax.add_patch(Polygon(trans, closed=True, facecolor=ORANGE, alpha=0.12, edgecolor=ORANGE, lw=2.2))
ax.text(0.45, 0.42, "area 1", color=ACCENT, fontsize=12)
ax.text(1.55, 1.35, "area 3", color=ORANGE, fontsize=12)
for v in trans[1:]:
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2, alpha=0.6))
ax.set_xlim(-0.5, 3.2)
ax.set_ylim(-0.5, 2.4)
ax.set_title("A = [[2, 1], [0, 1.5]],   det A = 3")
save(fig, "determinant")
