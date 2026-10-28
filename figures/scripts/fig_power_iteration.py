import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d, arrow

A = np.array([[2.0, 1.0], [1.0, 1.4]])
vals, vecs = np.linalg.eigh(A)
v1 = vecs[:, 1]
if v1[0] < 0:
    v1 = -v1

x = np.array([1.2, -0.5])
x = x / np.linalg.norm(x)
traj = [x]
for _ in range(7):
    x = A @ x
    x = x / np.linalg.norm(x)
    traj.append(x)
traj = np.array(traj)

fig, ax = plt.subplots(figsize=(6.4, 5.8))
axes2d(ax)
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), color=GRAY, lw=1.0, ls=":", alpha=0.8)
ax.plot([-1.25 * v1[0], 1.25 * v1[0]], [-1.25 * v1[1], 1.25 * v1[1]], "--", color=GREEN, lw=1.8)
ax.text(*(1.28 * v1 + np.array([0.03, -0.12])), r"$\mathbf{v}_1$", color=GREEN, fontsize=13)

ax.plot(traj[:, 0], traj[:, 1], color=ACCENT, lw=1.2, alpha=0.55)
start = np.array([199, 210, 254]) / 255
end = np.array([49, 46, 129]) / 255
for i, p in enumerate(traj):
    color = tuple(start + (end - start) * (i / (len(traj) - 1)))
    ax.plot(*p, "o", color=color, ms=7, mec="white", mew=0.9, zorder=5)

arrow(ax, (0, 0), traj[0], "#818cf8", lw=1.6)
arrow(ax, (0, 0), traj[-1], "#312e81", lw=2.2)
ax.text(*(traj[0] + np.array([0.05, -0.18])), r"$\mathbf{x}_0$", color="#6366f1", fontsize=12)
ax.text(*(traj[-1] + np.array([-0.16, 0.10])), r"$\mathbf{x}_7$", color="#312e81", fontsize=12)
ax.text(-1.18, -1.28, r"$r=|\lambda_2/\lambda_1|\approx0.24$", color=GRAY, fontsize=10)

ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.35)
ax.set_title("Power iteration: direction is pulled to $\\mathbf{v}_1$", loc="left")
save(fig, "power-iteration")
