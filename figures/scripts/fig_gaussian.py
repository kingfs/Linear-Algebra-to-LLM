import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, save

mu = np.array([0.4, -0.2])
Sigma = np.array([[1.4, 0.95], [0.95, 1.0]])
eigval, eigvec = np.linalg.eigh(Sigma)

x = np.linspace(-3.4, 3.4, 300)
y = np.linspace(-3.4, 3.4, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack([X, Y])
inv = np.linalg.inv(Sigma)
Z = np.exp(-0.5 * np.einsum("...i,ij,...j->...", pos - mu, inv, pos - mu))

fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.contourf(X, Y, Z, levels=8, cmap="Blues", alpha=0.55)
ax.contour(X, Y, Z, levels=5, colors="#7b8794", linewidths=0.8)
ax.plot(*mu, "o", color=ORANGE, ms=6)
ax.text(mu[0] + 0.12, mu[1] - 0.32, r"$\mu$", color=ORANGE, fontsize=13)

for i, color in enumerate([GREEN, ACCENT]):
    v = eigvec[:, i] * np.sqrt(eigval[i]) * 1.8
    ax.annotate("", xy=mu + v, xytext=mu, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))

ax.text(*(mu + eigvec[:, 0] * np.sqrt(eigval[0]) * 1.8 + np.array([0.08, 0.06])),
        r"$\sqrt{\lambda_1}$", color=GREEN, fontsize=11)
ax.text(*(mu + eigvec[:, 1] * np.sqrt(eigval[1]) * 1.8 + np.array([0.06, 0.06])),
        r"$\sqrt{\lambda_2}$", color=ACCENT, fontsize=11)

ax.set_xlim(-3.2, 3.8)
ax.set_ylim(-3.2, 3.4)
ax.set_aspect("equal")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_title("Bivariate Gaussian: elliptical level sets", loc="left")
save(fig, "gaussian-contours")
