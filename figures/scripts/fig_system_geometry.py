import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.2))
titles = ["Unique: one point", "None: parallel", "Infinite: same line"]
for ax, title in zip(axes, titles):
    st.axes2d(ax)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, loc="left", fontsize=11)

ax = axes[0]
xx = np.linspace(-0.5, 4.5, 2)
ax.plot(xx, 3 - xx, color=st.ACCENT, lw=2.0, label="x + y = 3")
ax.plot(xx, xx - 1, color=st.ORANGE, lw=2.0, label="x - y = 1")
ax.scatter([2], [1], color=st.RED, s=42, zorder=5)
ax.annotate("(2, 1)", xy=(2, 1), xytext=(2.35, 1.45), fontsize=10, color=st.RED)
ax.legend(loc="upper right", frameon=False, fontsize=8)

ax = axes[1]
ax.plot(xx, 3 - xx, color=st.ACCENT, lw=2.0, label="x + y = 3")
ax.plot(xx, 1 - xx, color=st.ORANGE, lw=2.0, label="x + y = 1")
ax.text(0.25, 3.75, "no common point", color=st.GRAY, fontsize=9)
ax.legend(loc="upper right", frameon=False, fontsize=8)

ax = axes[2]
ax.plot(xx, 3 - xx, color=st.ACCENT, lw=2.2, label="x + y = 3")
ax.plot(xx, 3 - xx, color=st.ORANGE, lw=6.0, alpha=0.22, label="2x + 2y = 6")
ax.text(0.25, 3.75, "the same line twice", color=st.GRAY, fontsize=9)
ax.legend(loc="upper right", frameon=False, fontsize=8)

fig.suptitle("A linear system is the intersection of its constraint lines", y=1.02)
fig.tight_layout()
st.save(fig, "system-geometry")
