import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save

rng = np.random.default_rng(7)
cov = np.array([[1.0, 0.82], [0.82, 1.0]])
X = rng.multivariate_normal([0, 0], cov, size=240)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]

fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.set_aspect("equal")
ax.grid(True, alpha=0.6)
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.scatter(X[:, 0], X[:, 1], s=8, color=ACCENT, alpha=0.35, edgecolors="none")
for i in range(2):
    v = vecs[:, i] * np.sqrt(vals[i]) * 2.2
    ax.annotate("", xy=v, xytext=(-v), arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=2.2))
    ax.text(*(v * 0.72 + np.array([0.06, 0.06])), rf"$v_{i+1}$", color=ORANGE, fontsize=14)
# 投影一个点
idx = int(np.argmax(X[:, 0] + X[:, 1]))
x = X[idx]
p = (x @ vecs[:, 0]) * vecs[:, 0]
ax.scatter([x[0]], [x[1]], color=GREEN, s=34, zorder=5)
ax.plot([x[0], p[0]], [x[1], p[1]], "--", color=GRAY, lw=1.2)
ax.scatter([p[0]], [p[1]], color=GREEN, s=28, zorder=5, marker="x")
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.2, 3.2)
ax.set_title("PCA: axes of maximum variance")
save(fig, "pca")
