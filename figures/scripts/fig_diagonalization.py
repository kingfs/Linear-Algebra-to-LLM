import numpy as np
import matplotlib.pyplot as plt
import _style as st

v1 = np.array([1.0, 0.0])
v2 = np.array([1.0, 1.0])
x = 0.7 * v1 + 0.9 * v2
Ax = 2.0 * 0.7 * v1 + 3.0 * 0.9 * v2

fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2))
ax = axes[0]
st.axes2d(ax)
ax.set_xlim(-0.4, 2.7)
ax.set_ylim(-0.65, 3.3)
ax.set_xlabel("standard x1")
ax.set_ylabel("standard x2")

st.arrow(ax, (0, 0), v1, st.ACCENT, lw=2.2, zorder=5)
st.arrow(ax, (0, 0), v2, st.GREEN, lw=2.2, zorder=5)
st.arrow(ax, (0, 0), x, st.ORANGE, lw=2.4, zorder=6)
c1 = 0.7 * v1
c2 = 0.9 * v2
ax.plot([c1[0], x[0]], [c1[1], x[1]], "--", color=st.GRAY, lw=1.0)
ax.plot([c2[0], x[0]], [c2[1], x[1]], "--", color=st.GRAY, lw=1.0)

ax.text(v1[0] + 0.05, v1[1] - 0.18, r"$v_1$", color=st.ACCENT, fontsize=12)
ax.text(v2[0] + 0.06, v2[1] + 0.04, r"$v_2$", color=st.GREEN, fontsize=12)
ax.text(x[0] + 0.06, x[1] + 0.06, r"$x$", color=st.ORANGE, fontsize=13)
ax.text(0.08, -0.20, r"$0.7v_1$", fontsize=8.5, color=st.GRAY)
ax.text(0.45, 0.62, r"$0.9v_2$", fontsize=8.5, color=st.GRAY)
ax.text(-0.38, -0.55, r"$x=0.7v_1+0.9v_2$", fontsize=9)
ax.set_title("Standard coordinates: mix the directions", loc="left", fontsize=11)

ax = axes[1]
st.axes2d(ax)
ax.set_xlim(-0.5, 4.9)
ax.set_ylim(-0.65, 3.6)
ax.set_xlabel("standard x1")
ax.set_ylabel("standard x2")

ax.plot([0, x[0]], [0, x[1]], ":", color=st.GRAY, lw=1.2, zorder=2)
ax.scatter([x[0]], [x[1]], color=st.GRAY, s=26, zorder=4)
ax.text(x[0] + 0.06, x[1] - 0.23, r"$x$", color=st.GRAY, fontsize=10)

st.arrow(ax, (0, 0), 1.4 * v1, st.ACCENT, lw=2.0, zorder=5)
st.arrow(ax, (0, 0), 2.7 * v2, st.GREEN, lw=2.0, zorder=5)
st.arrow(ax, (0, 0), Ax, st.ORANGE, lw=2.4, zorder=6)
a1 = 1.4 * v1
a2 = 2.7 * v2
ax.plot([a1[0], Ax[0]], [a1[1], Ax[1]], "--", color=st.GRAY, lw=1.0)
ax.plot([a2[0], Ax[0]], [a2[1], Ax[1]], "--", color=st.GRAY, lw=1.0)

ax.text(1.45, -0.22, r"$1.4v_1$", fontsize=8.5, color=st.ACCENT)
ax.text(2.78, 2.42, r"$2.7v_2$", fontsize=8.5, color=st.GREEN)
ax.text(Ax[0] + 0.06, Ax[1] + 0.06, r"$Ax$", color=st.ORANGE, fontsize=13)
ax.text(-0.42, -0.55, r"$Ax=1.4v_1+2.7v_2$", fontsize=9)
ax.text(3.05, -0.55, r"$D=\mathrm{diag}(2,3)$", fontsize=9, color=st.GRAY)
ax.set_title("Eigenbasis: D scales each coordinate", loc="left", fontsize=11)

fig.suptitle("Diagonalization: the same map is diagonal in the eigenbasis", y=1.02)
fig.tight_layout()
st.save(fig, "diagonalization")
