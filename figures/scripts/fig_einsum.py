import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import _style as st

fig, ax = plt.subplots(figsize=(9.4, 5.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.6)
ax.axis("off")

colors = {"b": "#94a3b8", "h": st.ACCENT, "i": st.GREEN, "j": st.ORANGE, "d": st.RED}


def block(x, y, letters, title, alpha=0.22):
    w, h = 3.2, 1.0
    ax.text(x + w / 2, y + h + 0.28, title, ha="center", va="bottom", fontsize=11)
    for k, letter in enumerate(letters):
        cell = Rectangle((x + k * w / len(letters), y), w / len(letters), h,
                         facecolor=colors[letter], edgecolor="white", alpha=alpha, lw=1.2)
        ax.add_patch(cell)
        ax.text(x + (k + 0.5) * w / len(letters), y + h / 2, letter,
                ha="center", va="center", fontsize=12, color=colors[letter], fontweight="bold")
    return x, y, w, h


qx, qy, qw, qh = block(1.0, 3.4, "bhid", "Q")
kx, ky, kw, kh = block(5.6, 3.4, "bhjd", "K")
sx, sy, sw, sh = block(3.3, 0.9, "bhij", "S = QK^T")

# shared b,h
ax.plot([qx + 0.2, qx + 0.2, kx + 0.2], [qy + qh + 0.05, 4.95, 4.95],
        color=st.GRAY, lw=1.2, ls=":")
ax.plot([qx + 0.95, kx + 0.95], [4.80, 4.80], color=st.GRAY, lw=0.8, ls=":")
ax.text(3.85, 5.02, "b,h shared", ha="center", fontsize=9, color=st.GRAY)

# contracted d
p1 = (qx + qw, qy + qh / 2)
p2 = (kx, ky + kh / 2)
ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<->", mutation_scale=12,
                             color=st.RED, lw=1.6, linestyle="--"))
ax.text((p1[0] + p2[0]) / 2, p1[1] + 0.22, "sum over d", ha="center", fontsize=9, color=st.RED)

# output keeps i and j
ax.add_patch(FancyArrowPatch((qx + 2 * qw / 4, qy), (sx + 2 * sw / 4, sy + sh),
                             arrowstyle="->", mutation_scale=10, color=st.GREEN, lw=1.0, ls=":"))
ax.add_patch(FancyArrowPatch((kx + 2 * kw / 4, ky), (sx + 3 * sw / 4, sy + sh),
                             arrowstyle="->", mutation_scale=10, color=st.ORANGE, lw=1.0, ls=":"))

ax.text(0.6, 2.45, "free: b, h, i, j", fontsize=10, color=st.GRAY)
ax.text(0.6, 2.15, "summed: d", fontsize=10, color=st.RED)
ax.text(0.6, 1.75, "same letter = same axis", fontsize=9, color=st.GRAY)
ax.text(0.6, 1.45, "missing letter = contracted", fontsize=9, color=st.GRAY)

ax.text(0.8, 0.20, "einsum: bhid,bhjd->bhij", fontsize=13, family="monospace")
ax.set_title("einsum: letters name axes; missing letters are summed", loc="left")
st.save(fig, "einsum")
