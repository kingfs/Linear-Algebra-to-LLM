import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import _style as st

fig, ax = plt.subplots(figsize=(11.2, 5.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.8)
ax.axis("off")


def box(x, y, w, h, title, sub, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=color, edgecolor=color, alpha=0.15, lw=1.4))
    ax.text(x + w / 2, y + h / 2 + 0.10, title, ha="center", va="center",
            fontsize=fs, color=color, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center",
            fontsize=fs - 1, color="#1f2933")


def arr(p, q, color=st.GRAY, lw=1.2, ls="-"):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=10))


box(0.3, 2.5, 1.3, 1.1, "X", "[T, D]", st.GRAY, fs=10)
box(2.0, 4.25, 1.0, 0.7, "$W_Q$", "", st.ACCENT, fs=9)
box(2.0, 2.95, 1.0, 0.7, "$W_K$", "", st.GREEN, fs=9)
box(2.0, 1.65, 1.0, 0.7, "$W_V$", "", st.ORANGE, fs=9)
box(3.5, 4.25, 1.5, 0.85, "Q", "[T, H, d]", st.ACCENT, fs=9)
box(3.5, 2.95, 1.5, 0.85, "K", "[T, H, d]", st.GREEN, fs=9)
box(3.5, 1.65, 1.5, 0.85, "V", "[T, H, d]", st.ORANGE, fs=9)

box(5.5, 3.55, 1.5, 0.95, "QK$^T$", "[H, T, T]", st.ACCENT, fs=9)
box(7.35, 3.55, 1.2, 0.95, "softmax", "[H, T, T]", st.GREEN, fs=8.5)
box(7.35, 1.70, 1.4, 0.95, "PV", "[H, T, d]", st.ORANGE, fs=9)
box(9.2, 1.70, 1.3, 0.95, "concat", "[T, D]", st.GRAY, fs=8.5)

arr((1.6, 3.05), (2.0, 4.60))
arr((1.6, 3.05), (2.0, 3.30))
arr((1.6, 3.05), (2.0, 2.00))
arr((3.0, 4.60), (3.5, 4.67))
arr((3.0, 3.30), (3.5, 3.37))
arr((3.0, 2.00), (3.5, 2.07))

arr((5.0, 4.67), (5.5, 4.22), st.ACCENT)
arr((5.0, 3.37), (5.5, 3.84), st.GREEN)
arr((7.0, 4.02), (7.35, 4.02), st.ACCENT)
arr((7.95, 3.55), (7.95, 2.65), st.GREEN)
arr((5.0, 2.07), (7.35, 2.17), st.ORANGE)
arr((8.75, 2.17), (9.2, 2.17), st.GRAY)

ax.text(0.35, 5.35, "QKV shapes: project, split heads, score, mix", fontsize=12,
        fontweight="bold")
ax.text(0.35, 0.95, r"$d=D/H$ per head; $H$ heads are batched as $[H,T,d]$", fontsize=9.5)
ax.text(0.35, 0.65, r"$QK^T$ scores one $T\times T$ matrix per head; softmax normalizes each query row",
        fontsize=8.5, color=st.GRAY)
ax.text(5.55, 4.70, r"$QK^T/\sqrt{d}$", fontsize=8, color=st.ACCENT)

ax.set_title("")
st.save(fig, "qkv-shapes")
