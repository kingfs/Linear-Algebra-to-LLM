import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(7.2, 5.2))
st.axes2d(ax)
ax.set_xlim(-0.45, 2.65)
ax.set_ylim(-0.62, 2.25)
ax.set_xlabel("x")
ax.set_ylabel("y")

x = np.array([2.0, 0.0])
y = np.array([1.5, 1.5])
p = np.array([1.0, 1.0])  # projection of x onto y

ax.plot([0, p[0]], [0, p[1]], color=st.GREEN, lw=6, alpha=0.22, zorder=1)
ax.plot([x[0], p[0]], [x[1], p[1]], color=st.GRAY, lw=1.2, ls="--", zorder=2)

st.arrow(ax, [0, 0], x, st.ACCENT, lw=2.4)
st.arrow(ax, [0, 0], y, st.ORANGE, lw=2.4)
st.arrow(ax, [0, 0], p, st.GREEN, lw=2.0, zorder=4)

# right-angle marker at the foot of the perpendicular
u = (x - p) / np.linalg.norm(x - p)
v = (np.array([0.0, 0.0]) - p) / np.linalg.norm(p)
s = 0.11
pts = [p + s * u, p + s * (u + v), p + s * v]
ax.plot([q[0] for q in pts], [q[1] for q in pts], color=st.GRAY, lw=1.0, zorder=4)

ax.scatter([p[0]], [p[1]], color=st.GREEN, s=30, zorder=5)
ax.text(2.08, -0.15, r"$\mathbf{x}$", color=st.ACCENT, fontsize=13)
ax.text(1.58, 1.22, r"$\mathbf{y}$", color=st.ORANGE, fontsize=13)
ax.text(1.02, 0.78, "shadow", color=st.GREEN, fontsize=10)
ax.text(0.18, 1.30, r"$\|\mathbf{y}\|=2.12$", color=st.ORANGE, fontsize=10)
ax.text(0.10, -0.48, r"$\mathbf{x}\cdot\mathbf{y}=3$", color=st.ACCENT, fontsize=11)
ax.text(0.10, -0.25, r"shadow $=\|\mathbf{x}\|\cos\theta=1.41$", color=st.GREEN, fontsize=10)

ax.set_title("Dot product = length of y times signed shadow of x", loc="left")
st.save(fig, "dot-product")
