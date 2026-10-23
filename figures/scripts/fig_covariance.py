import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, GRAY, save

rng = np.random.default_rng(7)
n = 46

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharex=True, sharey=True)

for ax, rho, title in [
    (axes[0], 0.82, "positive covariance"),
    (axes[1], -0.82, "negative covariance"),
]:
    cov = np.array([[1.0, rho], [rho, 1.0]])
    pts = rng.multivariate_normal([0, 0], cov, size=n)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=ACCENT, alpha=0.65, linewidths=0)
    ax.axhline(pts[:, 1].mean(), color=GRAY, lw=0.8, ls=":")
    ax.axvline(pts[:, 0].mean(), color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])

axes[0].set_ylabel("y")
axes[0].set_xlabel("x")
axes[1].set_xlabel("x")
fig.tight_layout()
save(fig, "covariance-sign")
