import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, GREEN, ORANGE, GRAY, save

u = np.array([2.4, 0.7])
v = np.array([0.9, 1.8])
s = u + v

fig, ax = plt.subplots(figsize=(4.8, 4.0))
ax.set_aspect("equal")
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.grid(True, alpha=0.5)

ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.4))
ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4))
ax.annotate("", xy=s, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.6))

ax.plot([v[0], s[0]], [v[1], s[1]], "--", color=GRAY, lw=1.2)
ax.plot([u[0], s[0]], [u[1], s[1]], "--", color=GRAY, lw=1.2)

ax.text(u[0] - 0.35, u[1] - 0.28, r"$\mathbf{u}$", color=ACCENT, fontsize=14)
ax.text(v[0] + 0.12, v[1] + 0.02, r"$\mathbf{v}$", color=GREEN, fontsize=14)
ax.text(s[0] + 0.12, s[1] + 0.02, r"$\mathbf{u}+\mathbf{v}$", color=ORANGE, fontsize=14)

ax.set_xlim(-0.4, 4.0)
ax.set_ylim(-0.4, 3.1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Tip-to-tail addition", loc="left")
save(fig, "vector-add")
