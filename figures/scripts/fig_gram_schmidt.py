import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(7.0, 5.4))
st.axes2d(ax)
ax.set_xlim(-0.35, 2.45)
ax.set_ylim(-0.35, 2.25)
ax.set_xlabel("x")
ax.set_ylabel("y")

v1 = np.array([2.0, 0.8])
v2 = np.array([1.1, 1.9])
t = (v2 @ v1) / (v1 @ v1)
p = t * v1
u2 = v2 - p

ax.plot([p[0], v2[0]], [p[1], v2[1]], color=st.GRAY, lw=1.2, ls="--", zorder=2)
st.arrow(ax, [0, 0], v1, st.ORANGE, lw=2.3)
st.arrow(ax, [0, 0], v2, st.ACCENT, lw=2.3)
st.arrow(ax, [0, 0], p, st.GREEN, lw=2.0, zorder=4)
st.arrow(ax, p, v2, st.RED, lw=2.1, ls="--", zorder=4)

# right angle between the projection and the residual
u = (v2 - p) / np.linalg.norm(v2 - p)
v = (np.array([0.0, 0.0]) - p) / np.linalg.norm(p)
s = 0.12
pts = [p + s * u, p + s * (u + v), p + s * v]
ax.plot([q[0] for q in pts], [q[1] for q in pts], color=st.GRAY, lw=1.0, zorder=5)

ax.text(1.90, 0.62, r"$\mathbf{v}_1$", color=st.ORANGE, fontsize=12)
ax.text(1.15, 1.82, r"$\mathbf{v}_2$", color=st.ACCENT, fontsize=12)
ax.text(1.56, 0.43, r"$\mathrm{proj}$", color=st.GREEN, fontsize=10)
ax.text(0.98, 1.38, r"$\mathbf{v}_2-\mathrm{proj}$", color=st.RED, fontsize=10)
ax.text(0.10, 2.02, r"$\mathbf{q}_1=\mathbf{v}_1/\|\mathbf{v}_1\|,\quad"
                    r"\mathbf{q}_2=(\mathbf{v}_2-\mathrm{proj})/\|\mathbf{v}_2-\mathrm{proj}\|$",
        fontsize=9.5)
ax.text(0.10, -0.25, "subtract the shadow, keep the perpendicular part", color=st.GRAY, fontsize=10)

ax.set_title("Gram-Schmidt: remove the shadow, keep the new direction", loc="left")
st.save(fig, "gram-schmidt")
