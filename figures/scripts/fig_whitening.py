import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from _style import ACCENT, ORANGE, GRAY, save

rng = np.random.default_rng(11)
n = 220
Sigma = np.array([[2.2, 1.5], [1.5, 1.2]])
X = rng.multivariate_normal([0, 0], Sigma, size=n)

eigval, eigvec = np.linalg.eigh(Sigma)
D_inv = np.diag(1.0 / np.sqrt(eigval))
W_pca = D_inv @ eigvec.T
W_zca = eigvec @ D_inv @ eigvec.T
Y_pca = X @ W_pca.T
Y_zca = X @ W_zca.T

panels = [
    (X, "original: tilted ellipse", ORANGE),
    (Y_pca, "PCA whitening: rotate, then scale", ACCENT),
    (Y_zca, "ZCA whitening: scale in place", ACCENT),
]

fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharex=True, sharey=True)
for ax, (pts, title, color) in zip(axes, panels):
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.45, linewidths=0)
    ax.axhline(0, color=GRAY, lw=0.7, ls=":")
    ax.axvline(0, color=GRAY, lw=0.7, ls=":")
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=11.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-4.6, 4.6)

ax = axes[0]
ax.add_patch(
    Ellipse((0, 0), 2 * np.sqrt(eigval[1]) * 2, 2 * np.sqrt(eigval[0]) * 2,
            angle=np.degrees(np.arctan2(eigvec[1, 1], eigvec[0, 1])),
            fill=False, color=ORANGE, lw=1.2)
)
for ax in axes[1:]:
    ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, color=ACCENT, lw=1.2))

fig.tight_layout()
save(fig, "whitening")
