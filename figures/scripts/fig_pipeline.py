"""One token's journey through a decoder-only transformer, as a block diagram."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _style import ACCENT, GREEN, ORANGE, save

INK = "#1f2933"

fig, ax = plt.subplots(figsize=(6.4, 7.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.axis("off")

X0, W = 1.55, 6.9
CENTER = X0 + W / 2


def box(y, h, text, fc, ec, fs=11.5, weight="normal", dashed=False, rounding=1.0):
    patch = FancyBboxPatch(
        (X0, y - h / 2), W, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=1.1, edgecolor=ec, facecolor=fc,
        linestyle=(0, (4, 2)) if dashed else "solid",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(CENTER, y, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight=weight, linespacing=1.55, zorder=3)
    return patch


def arrow(y_from, y_to):
    ax.annotate("", xy=(CENTER, y_to), xytext=(CENTER, y_from),
                arrowprops=dict(arrowstyle="-|>", color=GRAY_MID, lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=11),
                zorder=1)


GRAY_MID = "#94a3b8"
INDIGO_BG, INDIGO_ED = "#eef2ff", "#c7d2fe"
ORANGE_BG, ORANGE_ED = "#fff7ed", "#fdba74"
GREEN_BG, GREEN_ED = "#ecfdf5", "#99f6e4"

nodes = [
    (94.0, 7.0, 'text  "cat"', "#f8fafc", "#cbd5e1", 11.5, "normal"),
    (81.0, 7.0, "Token ID  8721", INDIGO_BG, INDIGO_ED, 11.5, "normal"),
    (68.0, 7.4, "Embedding   $x\\in\\mathbb{R}^{4096}$", INDIGO_BG, INDIGO_ED, 11.5, "normal"),
    (55.0, 7.0, "+ Positional encoding", INDIGO_BG, INDIGO_ED, 11.5, "normal"),
    (37.0, 15.0, "", ORANGE_BG, ORANGE_ED, 11.5, "normal"),
    (19.0, 7.0, "Logits   $g\\in\\mathbb{R}^{V}$", INDIGO_BG, INDIGO_ED, 11.5, "normal"),
    (6.0, 7.0, "Softmax   $p\\in\\Delta^{V}$", GREEN_BG, GREEN_ED, 11.5, "normal"),
]

for y, h, text, fc, ec, fs, weight in nodes:
    box(y, h, text, fc, ec, fs=fs, weight=weight)

# transformer-block internals
ax.text(CENTER, 41.6, "Transformer block   × L", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=INK, zorder=3)
ax.text(CENTER, 37.2, "RMSNorm → Attention → + residual", ha="center", va="center",
        fontsize=9.6, color=INK, zorder=3)
ax.text(CENTER, 33.4, "RMSNorm → FFN → + residual", ha="center", va="center",
        fontsize=9.6, color=INK, zorder=3)

# arrows between stacked nodes
pairs = [(90.5, 84.5), (77.5, 71.7), (64.3, 58.5), (51.5, 44.5), (29.5, 22.5), (15.5, 9.5)]
for a, b in pairs:
    arrow(a, b)

# highlight: the block is repeated L times
ax.annotate("", xy=(X0 + W + 0.25, 44.5), xytext=(X0 + W + 0.25, 29.5),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2, shrinkA=0, shrinkB=0))
ax.text(X0 + W + 0.55, 37.0, "L layers", rotation=90, ha="left", va="center",
        fontsize=10, color=ORANGE, fontweight="bold")

save(fig, "transformer-pipeline")
