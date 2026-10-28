import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(5.4, 7.6))
st.axes2d(ax)
ax.set_xlim(-0.45, 4.65)
ax.set_ylim(-0.45, 7.55)
ax.set_xlabel("x")
ax.set_ylabel("y")

a1 = np.array([2.0, 1.0])
a2 = np.array([1.0, 3.0])
x = np.array([1.0, 2.0])
out = x[0] * a1 + x[1] * a2

poly = np.array([[0, 0], x[0] * a1, out, x[1] * a2])
ax.fill(poly[:, 0], poly[:, 1], color=st.ACCENT, alpha=0.07, zorder=1)
ax.plot([x[0] * a1[0], out[0]], [x[0] * a1[1], out[1]], color=st.GRAY, lw=1.1, ls="--", zorder=2)
ax.plot([x[1] * a2[0], out[0]], [x[1] * a2[1], out[1]], color=st.GRAY, lw=1.1, ls="--", zorder=2)

st.arrow(ax, [0, 0], a1, st.ORANGE, lw=2.3)
st.arrow(ax, [0, 0], x[1] * a2, st.GREEN, lw=2.3, ls="--")
st.arrow(ax, [0, 0], out, st.ACCENT, lw=2.7)

ax.text(2.06, 0.72, r"$\mathbf{a}_1=(2,1)$", color=st.ORANGE, fontsize=10)
ax.text(1.98, 5.78, r"$2\mathbf{a}_2=(2,6)$", color=st.GREEN, fontsize=10)
ax.text(3.76, 6.96, r"$A\mathbf{x}=(4,7)$", color=st.ACCENT, fontsize=10)
ax.text(0.12, -0.35, r"$\mathbf{x}=(1,2)$: take one $\mathbf{a}_1$ and two $\mathbf{a}_2$", fontsize=10)
ax.text(0.12, 7.22, "A = [[2, 1], [1, 3]]", fontsize=10)

ax.set_title("Ax = weighted sum of the columns", loc="left")
st.save(fig, "matrix-vector-columns")
