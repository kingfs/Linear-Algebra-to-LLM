import numpy as np
import matplotlib.pyplot as plt
import _style as st

b1 = np.array([2.0, 0.2])
b2 = np.array([-0.2, 1.8])
b3 = b1 + b2
v = 1.1 * b1 + 0.7 * b2

fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0))

for ax in axes:
    st.axes2d(ax)
    ax.set_xlim(-0.7, 3.3)
    ax.set_ylim(-0.8, 2.7)
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")

ax = axes[0]
par = np.array([[0, 0], b1, b1 + b2, b2])
ax.fill(par[:, 0], par[:, 1], color=st.ACCENT, alpha=0.07, zorder=1)
st.arrow(ax, (0, 0), b1, st.ACCENT, lw=2.2, zorder=5)
st.arrow(ax, (0, 0), b2, st.GREEN, lw=2.2, zorder=5)
st.arrow(ax, (0, 0), v, st.ORANGE, lw=2.4, zorder=6)

c1 = 1.1 * b1
c2 = 0.7 * b2
ax.plot([c1[0], v[0]], [c1[1], v[1]], "--", color=st.GRAY, lw=1.0, zorder=2)
ax.plot([c2[0], v[0]], [c2[1], v[1]], "--", color=st.GRAY, lw=1.0, zorder=2)

ax.text(b1[0] + 0.05, b1[1] + 0.08, r"$b_1$", color=st.ACCENT, fontsize=12)
ax.text(b2[0] - 0.18, b2[1] + 0.08, r"$b_2$", color=st.GREEN, fontsize=12)
ax.text(v[0] + 0.05, v[1] + 0.08, r"$v$", color=st.ORANGE, fontsize=12)
ax.text(1.25, 0.06, r"$1.1b_1$", fontsize=8.5, color=st.GRAY)
ax.text(-0.15, 1.28, r"$0.7b_2$", fontsize=8.5, color=st.GRAY)
ax.text(-0.62, 2.48, r"$\mathcal{B}=\{b_1,b_2\}$", fontsize=10)
ax.text(-0.62, 2.25, r"$\dim\,\mathrm{span}=2$", fontsize=9, color=st.ACCENT)
ax.text(-0.62, -0.68, r"$v=1.1b_1+0.7b_2$", fontsize=9)
ax.set_title("Basis: exactly enough directions", loc="left", fontsize=11)

ax = axes[1]
par = np.array([[0, 0], b1, b1 + b2, b2])
ax.fill(par[:, 0], par[:, 1], color=st.ACCENT, alpha=0.05, zorder=1)
st.arrow(ax, (0, 0), b1, st.ACCENT, lw=2.0, zorder=5)
st.arrow(ax, (0, 0), b2, st.GREEN, lw=2.0, zorder=5)
st.arrow(ax, (0, 0), b3, st.GRAY, lw=1.8, ls="--", zorder=4)
st.arrow(ax, (0, 0), v, st.ORANGE, lw=2.4, zorder=6)

ax.text(b1[0] + 0.05, b1[1] + 0.08, r"$b_1$", color=st.ACCENT, fontsize=12)
ax.text(b2[0] - 0.18, b2[1] + 0.08, r"$b_2$", color=st.GREEN, fontsize=12)
ax.text(b3[0] - 0.08, b3[1] + 0.08, r"$b_3$", color=st.GRAY, fontsize=12)
ax.text(v[0] + 0.05, v[1] + 0.08, r"$v$", color=st.ORANGE, fontsize=12)
ax.text(-0.62, 2.48, r"$b_3=b_1+b_2$", fontsize=10, color=st.GRAY)
ax.text(-0.62, 2.25, r"still $\dim=2$", fontsize=9, color=st.ACCENT)
ax.text(-0.62, -0.55, r"$v=0.4b_1+0.7b_3$", fontsize=9)
ax.text(-0.62, -0.76, "coordinates are not unique", fontsize=8.5, color=st.RED)
ax.set_title("Add a dependent vector: still dimension 2", loc="left", fontsize=11)

fig.suptitle("Basis = enough directions, no redundancy", y=1.02)
fig.tight_layout()
st.save(fig, "basis-dimension")
