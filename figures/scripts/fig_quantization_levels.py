import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save

s = 1.0 / 7.0
ks = np.arange(-7, 8)
vals = np.array([0.37, -0.62, 0.08, 0.83])
q = np.round(vals / s) * s

fig, ax = plt.subplots(figsize=(9.6, 3.8))
ax.set_xlim(-1.12, 1.12)
ax.set_ylim(-0.92, 0.92)
ax.axhline(0, color="#c9c2b2", lw=1.0)
for k in ks:
    ax.plot([k * s, k * s], [-0.09, 0.09], color=GRAY, lw=0.9)
for v in [-1.0, 0.0, 1.0]:
    ax.text(v, -0.21, f"{v:g}", ha="center", va="top", color="#52606d", fontsize=10)

ax.scatter(vals, np.full_like(vals, 0.38), s=54, color=ACCENT, zorder=5, label="continuous x")
ax.scatter(q, np.full_like(q, -0.38), s=54, color=ORANGE, zorder=5, label="quantized q")
for v, qi in zip(vals, q):
    ax.annotate(
        "",
        xy=(qi, -0.29),
        xytext=(v, 0.29),
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1, ls="--", shrinkA=0, shrinkB=0),
    )
    ax.text(v, 0.49, f"{v:+.2f}", color=ACCENT, fontsize=9, ha="center")
    ax.text(qi, -0.55, f"{qi:+.2f}", color=ORANGE, fontsize=9, ha="center")

# 相邻刻度之间的步长 s
ax.annotate(
    "",
    xy=(s, 0.70),
    xytext=(0.0, 0.70),
    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.1, shrinkA=0, shrinkB=0),
)
ax.text(s / 2, 0.76, r"$s=1/7\approx0.143$", color=ORANGE, fontsize=10, ha="center")

# 最大误差的图示
v0, q0 = vals[0], q[0]
ax.annotate(
    "",
    xy=(q0, -0.72),
    xytext=(v0, -0.72),
    arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.2, shrinkA=0, shrinkB=0),
)
ax.text(q0 + 0.12, -0.79, r"$|x-q|\leq s/2$", color=GREEN, fontsize=9)

ax.set_yticks([])
ax.set_xticks([])
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False, fontsize=10)
ax.set_title("Quantization: snap each value to the nearest grid point", loc="left")
save(fig, "quantization-levels")
