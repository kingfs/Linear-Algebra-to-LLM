import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d, arrow

A = np.diag([1.0, 20.0])
x0 = np.array([1.5, 0.45])
eta = 2.0 / (A[0, 0] + A[1, 1])

traj = [x0]
x = x0.copy()
for _ in range(16):
    x = x - eta * (A @ x)
    traj.append(x)
traj = np.array(traj)

xg = np.linspace(-2.0, 2.0, 260)
yg = np.linspace(-1.3, 1.3, 260)
X, Y = np.meshgrid(xg, yg)
Z = 0.5 * (A[0, 0] * X**2 + A[1, 1] * Y**2)

fig, ax = plt.subplots(figsize=(8.6, 5.0))
axes2d(ax, grid=False)
ax.imshow(
    Z,
    extent=[xg[0], xg[-1], yg[0], yg[-1]],
    origin="lower",
    cmap="Blues",
    alpha=0.45,
    interpolation="bilinear",
    zorder=0,
)
levels = np.linspace(0.03, 12.0, 11)
ax.contour(X, Y, Z, levels=levels, colors=GRAY, linewidths=0.6, alpha=0.75)

ax.plot(traj[:, 0], traj[:, 1], color=ORANGE, lw=1.5, alpha=0.8)
ax.plot(traj[:, 0], traj[:, 1], "o", color=ORANGE, ms=4.5, alpha=0.9)
ax.plot(*x0, "o", color=ACCENT, ms=8, zorder=5)
ax.text(x0[0] + 0.05, x0[1] + 0.10, r"$\mathbf{x}_0$", color=ACCENT, fontsize=12)

arrow(ax, x0, np.array([0.0, 0.0]), GREEN, lw=2.4, ls="--")
ax.plot(0, 0, "*", color=GREEN, ms=12, zorder=6)
ax.text(0.18, -0.24, "Newton: one step", color=GREEN, fontsize=11)
ax.text(-1.82, 0.92, "GD: zigzag along the valley", color=ORANGE, fontsize=10)

ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-1.1, 1.1)
ax.set_title("Same narrow bowl: GD zigzags, Newton jumps to the center", loc="left")
save(fig, "gradient-descent")
