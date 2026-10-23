import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, save

A = np.array([[2.0, 1.0], [0.0, 1.0]])
U, s, Vt = np.linalg.svd(A)
t = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(t), np.sin(t)])
ellipse = A @ circle

fig, ax = plt.subplots(figsize=(5.4, 4.8))
ax.set_aspect("equal")
ax.grid(True, alpha=0.65)
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.plot(circle[0], circle[1], color=ACCENT, lw=2.0, label="unit circle")
ax.plot(ellipse[0], ellipse[1], color=ORANGE, lw=2.4, label="A · circle")
for i in range(2):
    v = s[i] * U[:, i]
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
    ax.text(*(v * 0.62 + np.array([0.06, 0.06])), rf"$\sigma_{i+1}={s[i]:.2f}$", color=GREEN, fontsize=12)
ax.set_xlim(-2.8, 2.8)
ax.set_ylim(-1.8, 2.6)
ax.legend(loc="lower left", frameon=False)
ax.set_title("SVD: the unit circle becomes an ellipse")
save(fig, "svd")
