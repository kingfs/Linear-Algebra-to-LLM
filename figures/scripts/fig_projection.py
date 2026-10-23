import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save

x = np.array([3.0, 1.0])
u = np.array([0.8, 0.6])
p = (x @ u) * u
r = x - p

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.set_aspect("equal")
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.grid(True, alpha=0.7)

ax.annotate("", xy=x, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.4))
ax.text(*(x + np.array([0.08, 0.1])), r"$\mathbf{x}$", color=ACCENT, fontsize=14)
ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
ax.text(*(u + np.array([0.06, 0.06])), r"$\mathbf{u}$", color=GREEN, fontsize=14)
ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0))
ax.text(*(p + np.array([0.03, -0.22])), r"$\mathbf{p}$", color=ORANGE, fontsize=14)
ax.plot([p[0], x[0]], [p[1], x[1]], "--", color=GRAY, lw=1.4)
ax.text(*(p + r * 0.5 + np.array([0.06, 0.05])), r"$\mathbf{x}-\mathbf{p}$", color=GRAY, fontsize=12)
ax.set_xlim(-0.5, 3.4)
ax.set_ylim(-0.6, 1.8)
ax.set_title("Projection of x onto the direction u")
save(fig, "projection")
