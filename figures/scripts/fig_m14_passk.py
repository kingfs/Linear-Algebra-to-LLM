"""M14 figure: pass@k vs pass^k at the same per-trial success rate."""

import numpy as np
import matplotlib.pyplot as plt

from _style import ACCENT, ORANGE, GRAY, save

k = np.arange(1, 11)

fig, ax = plt.subplots(figsize=(6.6, 4.4))

for p, color in ((0.5, ACCENT), (0.8, ORANGE)):
    at_least_one = 1 - (1 - p) ** k
    all_k = p ** k
    ax.plot(k, at_least_one, "-o", color=color, lw=2.0, ms=4,
            label=fr"pass@k, p={p:.1f}")
    ax.plot(k, all_k, "--s", color=color, lw=1.8, ms=4,
            label=fr"pass^k, p={p:.1f}")

# Highlight the k=4, p=0.5 pair.
ax.annotate("", xy=(4, 0.5 ** 4), xytext=(4, 1 - 0.5 ** 4),
            arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.4))
ax.text(4.12, 0.50, "93.8%\nvs 6.25%", fontsize=8.5, color=GRAY, va="center")

ax.set_xlabel("k (number of attempts)")
ax.set_ylabel("probability")
ax.set_ylim(-0.02, 1.04)
ax.set_xticks(k)
ax.grid(True, alpha=0.55)
ax.set_title("pass@k vs pass^k: at least once vs every time", loc="left")

ax.text(0.98, 0.04,
        r"$\mathrm{pass@}k=1-(1-p)^k$" + "\n" + r"$\mathrm{pass}^k=p^k$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        color="#1f2933", bbox=dict(boxstyle="round,pad=0.35", fc="#fffdf8",
                                   ec="#e6e0d4"))

fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4,
           frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(bottom=0.24)
save(fig, "m14-passk")
