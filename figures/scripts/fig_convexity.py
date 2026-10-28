import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save, axes2d, arrow

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))

# ---------------------------------------------------------------- 左：凸
ax = axes[0]
x = np.linspace(-1.55, 1.55, 400)
f = x**2
ax.plot(x, f, color=ACCENT, lw=2.4)
a, b = -1.2, 1.2
ax.plot([a, b], [a**2, b**2], color=ORANGE, lw=1.6, ls="--")
xf = np.linspace(a, b, 200)
chord = a**2 + (b**2 - a**2) * (xf - a) / (b - a)
ax.fill_between(xf, xf**2, chord, color=ORANGE, alpha=0.15)
ax.plot([a, b], [a**2, b**2], "o", color=ORANGE, ms=6)
ax.text(-1.48, 2.24, r"chord above graph", color=ORANGE, fontsize=10)
ax.text(-1.48, 1.96, r"$f(\theta x+(1-\theta)y)$", color=GRAY, fontsize=9)
ax.set_title("Convex: chord stays above the graph", loc="left")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-0.25, 2.6)

# ---------------------------------------------------------------- 右：非凸
ax = axes[1]
f = x**4 - 2 * x**2
ax.plot(x, f, color=GREEN, lw=2.4)
a, b = -1.4, 1.4
fa, fb = a**4 - 2 * a**2, b**4 - 2 * b**2
ax.plot([a, b], [fa, fb], color=ORANGE, lw=1.6, ls="--")
xf = np.linspace(a, b, 200)
chord = fa + (fb - fa) * (xf - a) / (b - a)
ax.fill_between(xf, xf**4 - 2 * xf**2, chord, where=(xf**4 - 2 * xf**2) >= chord, color=ORANGE, alpha=0.18)
ax.plot([a, b], [fa, fb], "o", color=ORANGE, ms=6)
ax.plot([-1, 1], [-1, -1], "o", color=GREEN, ms=5)
ax.text(-1.48, 1.26, "graph rises above chord", color=GREEN, fontsize=10)
ax.text(-0.30, 0.20, "violation", color=ORANGE, fontsize=10)
ax.text(-1.48, 0.96, "local max at 0", color=GRAY, fontsize=9)
ax.set_title("Not convex: graph can rise above the chord", loc="left")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.25, 1.65)

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.4)

fig.subplots_adjust(wspace=0.22)
save(fig, "convexity")
