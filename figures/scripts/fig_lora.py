import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import _style as st

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))


def box(ax, xy, w, h, text, color, alpha=0.16, fs=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=color, edgecolor=color, alpha=alpha, lw=1.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arr(ax, p, q, color=st.GRAY, lw=1.3, ls="-"):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))


for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("", loc="left")

ax = axes[0]
box(ax, (2.6, 2.2), 3.4, 1.6, "W\nD x D", st.GRAY, fs=10)
ax.text(0.4, 2.9, "x", fontsize=12)
arr(ax, (1.0, 3.0), (2.55, 3.0))
arr(ax, (6.05, 3.0), (7.4, 3.0))
ax.text(7.55, 2.9, "y", fontsize=12)
ax.text(2.6, 1.55, "all D^2 = 16.8M entries trainable", fontsize=9, color=st.GRAY)
ax.set_title("Full fine-tuning: rewrite W", loc="left", fontsize=11)

ax = axes[1]
box(ax, (4.1, 3.8), 4.0, 1.2, "W\nD x D  (frozen)", st.GRAY, fs=9)
ax.text(0.4, 4.4, "x", fontsize=12)
arr(ax, (1.0, 4.4), (4.05, 4.4))
arr(ax, (8.15, 4.4), (8.75, 4.4))
ax.add_patch(Circle((8.95, 4.4), 0.22, facecolor=st.ACCENT, edgecolor=st.ACCENT, alpha=0.18))
ax.text(8.95, 4.4, "+", ha="center", va="center", fontsize=14, color=st.ACCENT)
arr(ax, (9.2, 4.4), (9.75, 4.4))
ax.text(9.82, 4.3, "y", fontsize=12)

box(ax, (2.0, 1.5), 2.2, 1.1, "A\nr x D", st.GREEN, fs=9)
box(ax, (5.3, 1.5), 2.2, 1.1, "B\nD x r", st.RED, fs=9)
arr(ax, (1.0, 4.15), (1.0, 2.05), color=st.GREEN, ls="--")
arr(ax, (1.0, 2.05), (1.95, 2.05), color=st.GREEN, ls="--")
arr(ax, (4.25, 2.05), (5.25, 2.05), color=st.RED, ls="--")
arr(ax, (7.55, 2.05), (8.95, 4.15), color=st.RED, ls="--")
ax.text(2.0, 2.85, "down-project", fontsize=8, color=st.GREEN)
ax.text(5.3, 2.85, "up-project", fontsize=8, color=st.RED)
ax.text(1.7, 0.85, "2Dr = 131,072 trainable  (0.78% of D^2)", fontsize=9, color=st.GRAY)
ax.set_title("LoRA: freeze W, learn a rank-r update", loc="left", fontsize=11)

fig.suptitle("LoRA replaces a full matrix update with a low-rank bottleneck", y=1.02)
fig.tight_layout()
st.save(fig, "lora")
