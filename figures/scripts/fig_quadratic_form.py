import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d, arrow

A = np.array([[2.0, 0.5], [0.5, 1.2]])
B = np.array([[1.0, 0.0], [0.0, -1.0]])
x = np.linspace(-2.2, 2.2, 260)
y = np.linspace(-2.2, 2.2, 260)
X, Y = np.meshgrid(x, y)


def qform(M):
    return M[0, 0] * X**2 + 2 * M[0, 1] * X * Y + M[1, 1] * Y**2


fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.7))
for ax, M, title in [
    (axes[0], A, "Positive definite: a bowl (all $\\lambda>0$)"),
    (axes[1], B, "Indefinite: a saddle ($\\lambda$ signs differ)"),
]:
    axes2d(ax, grid=False)
    Z = qform(M)
    vmax = np.abs(Z).max()
    ax.imshow(
        Z,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.65,
        interpolation="bilinear",
        zorder=0,
    )
    levels = np.linspace(-vmax, vmax, 9)
    ax.contour(X, Y, Z, levels=levels, colors=GRAY, linewidths=0.6, alpha=0.85)
    ax.axhline(0, color="#c9c2b2", lw=0.7)
    ax.axvline(0, color="#c9c2b2", lw=0.7)

    vals, vecs = np.linalg.eigh(M)
    for i, color in enumerate([ACCENT, GREEN]):
        v = vecs[:, i]
        if v[0] < 0:
            v = -v
        arrow(ax, (0, 0), 1.7 * v, color, lw=1.8)
        ax.text(*(1.78 * v + np.array([0.05, 0.05])), rf"$\lambda_{i + 1}={vals[i]:.1f}$", color=color, fontsize=10)

    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_title(title, loc="left")

axes[0].text(-1.95, 1.72, r"$Q(\mathbf{x})>0$", color=ACCENT, fontsize=11)
axes[1].text(1.18, 0.32, r"$Q>0$", color="#b91c1c", fontsize=11)
axes[1].text(0.28, 1.62, r"$Q<0$", color="#1d4ed8", fontsize=11)
axes[1].contour(X, Y, qform(B), levels=[0], colors="#111827", linewidths=1.2)

fig.subplots_adjust(wspace=0.16)
save(fig, "quadratic-form")
