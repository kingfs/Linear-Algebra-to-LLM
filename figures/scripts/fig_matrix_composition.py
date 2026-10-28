import numpy as np
import matplotlib.pyplot as plt
import _style as st

ang = np.deg2rad(25)
B = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
A = np.array([[1.0, 0.7], [0.0, 1.0]])
AB = A @ B

square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
bs = square @ B.T
absq = bs @ A.T
ab = square @ AB.T


def draw(ax, poly, color, label, alpha=0.12, lw=2.0, ls="-"):
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, zorder=1)
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, ls=ls, label=label, zorder=3)


fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7))
for ax in axes:
    st.axes2d(ax)
    ax.set_xlim(-1.05, 2.05)
    ax.set_ylim(-1.05, 2.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

ax = axes[0]
draw(ax, square, st.GRAY, "unit square", alpha=0.06)
draw(ax, bs, st.ORANGE, "B(square)", alpha=0.10)
draw(ax, absq, st.ACCENT, "A(B(square))", alpha=0.12)
ax.annotate("", xy=(0.95, 0.95), xytext=(0.05, 0.05),
            arrowprops=dict(arrowstyle="->", color=st.GRAY, lw=1.2))
ax.text(0.22, 1.02, "B", color=st.ORANGE, fontsize=10)
ax.text(0.80, 1.62, "then A", color=st.ACCENT, fontsize=10)
ax.set_title("Step by step: first B, then A", loc="left")
ax.legend(loc="lower left", frameon=False, fontsize=8)

ax = axes[1]
draw(ax, square, st.GRAY, "unit square", alpha=0.06)
draw(ax, ab, st.ACCENT, "AB(square)", alpha=0.12)
ax.text(0.10, 1.55, r"$AB=A\,B$", fontsize=11)
ax.text(0.10, 1.28, "same final map, one matrix", color=st.GRAY, fontsize=9)
ax.set_title("One matrix: AB", loc="left")
ax.legend(loc="lower left", frameon=False, fontsize=8)

fig.suptitle("Composition: apply B, then A; the product AB does both", y=1.02)
fig.tight_layout()
st.save(fig, "matrix-composition")
