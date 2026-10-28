import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(7.0, 6.4))
st.axes2d(ax)
ax.set_xlim(-1.75, 1.75)
ax.set_ylim(-1.75, 1.75)
ax.set_xlabel("x")
ax.set_ylabel("y")

theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), color=st.ACCENT, lw=2.0)

for c in [-1.2, -0.4, 0.4, 1.0, np.sqrt(2)]:
    xs = np.linspace(-1.7, 1.7, 2)
    ax.plot(xs, c - xs, color=st.GRAY, lw=0.8, alpha=0.55)

p = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
q = -p
ax.scatter([p[0], q[0]], [p[1], q[1]], color=[st.RED, st.GRAY], s=[50, 30], zorder=6)
ax.plot([-1.7, 1.7], [np.sqrt(2) + 1.7, np.sqrt(2) - 1.7],
        color=st.ORANGE, lw=2.2, zorder=4)

# objective gradient and constraint normal at the maximum
gf = np.array([1.0, 1.0])
gg = np.array([2 * p[0], 2 * p[1]])
st.arrow(ax, p, p + 0.42 * gf, st.RED, lw=2.2, zorder=7)
st.arrow(ax, p, p + 0.28 * gg, st.GREEN, lw=2.0, ls="--", zorder=7)

ax.text(0.72, 0.93, r"$\nabla f$", color=st.RED, fontsize=11)
ax.text(1.03, 1.30, r"$\nabla g$", color=st.GREEN, fontsize=11)
ax.text(-1.62, 1.53, r"$f=x+y$", fontsize=11)
ax.text(-1.62, 1.32, r"$g=x^2+y^2-1=0$", fontsize=10)
ax.text(-1.62, -1.58, r"at the max: $\nabla f=\lambda\nabla g$", fontsize=10)
ax.text(1.02, -1.48, "constraint", color=st.ACCENT, fontsize=9)
ax.text(0.20, 1.38, "tangent level line", color=st.ORANGE, fontsize=9)

ax.set_title("Lagrange multiplier: the two gradients are parallel", loc="left")
st.save(fig, "lagrange")
