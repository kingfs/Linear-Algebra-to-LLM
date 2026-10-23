import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save

theta = 0.38
fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.set_aspect("equal")
ax.grid(True, alpha=0.55)
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
colors = [GRAY, ACCENT, ORANGE, GREEN]
for pos in range(4):
    ang = pos * theta
    v = np.array([np.cos(ang), np.sin(ang)])
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colors[pos], lw=2.2))
    ax.text(*(v * 1.08), f"pos {pos}", color=colors[pos], fontsize=11)
    arc = np.linspace(0, ang, 50)
    ax.plot(0.38 * np.cos(arc), 0.38 * np.sin(arc), color=colors[pos], lw=1.0, alpha=0.6)
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.35)
ax.set_title(r"RoPE: rotate by position $\times\ \theta$")
save(fig, "rope")
