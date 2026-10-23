import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, save

A = np.array([[2.0, 1.0], [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)

fig, ax = plt.subplots(figsize=(5.4, 4.8))
ax.set_aspect("equal")
ax.grid(True, alpha=0.65)
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
def grid(ax, A, color, alpha, lw):
    for t in np.arange(-2, 2.01, 0.5):
        line = np.array([[-2, t], [2, t]]) @ A.T
        ax.plot(line[:, 0], line[:, 1], color=color, lw=lw, alpha=alpha)
        line = np.array([[t, -2], [t, 2]]) @ A.T
        ax.plot(line[:, 0], line[:, 1], color=color, lw=lw, alpha=alpha)


grid(ax, np.eye(2), "#c9c2b2", 0.7, 0.7)
grid(ax, A, ACCENT, 0.35, 1.0)
for val, vec in zip(vals, vecs.T):
    ax.annotate("", xy=val * vec, xytext=(-val * vec), arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=2.0))
    ax.text(*(val * vec * 0.55 + np.array([0.08, 0.08])), rf"$\lambda={val:.0f}$", color=ORANGE, fontsize=12)
ax.set_xlim(-4.2, 4.2)
ax.set_ylim(-4.2, 4.2)
ax.set_title("Eigen-directions stay on their line after A")
save(fig, "eigenvectors")
