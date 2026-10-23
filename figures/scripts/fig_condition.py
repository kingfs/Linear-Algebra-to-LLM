import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from _style import ACCENT, ORANGE, GRAY, save

sig_max, sig_min = 2.0, 0.25
theta = np.radians(28)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

# ---- 左：单位圆，两个挨得很近的输入
ax = axes[0]
ax.add_patch(Circle((0, 0), 1.0, fill=False, color=ACCENT, lw=2.0))
d = np.radians(5)
pts = np.array([[np.cos(d), np.sin(d)], [np.cos(d), -np.sin(d)]])
ax.plot(pts[:, 0], pts[:, 1], "o", color=ORANGE, ms=6)
ax.annotate("", xy=pts[0], xytext=pts[1], arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.2))
ax.text(1.12, 0.0, r"$\delta$", color=GRAY, fontsize=12)
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.set_title("input: unit circle", loc="left")
ax.set_xticks([])
ax.set_yticks([])

# ---- 右：椭圆，两个像几乎重合
ax = axes[1]
ax.add_patch(
    Ellipse((0, 0), 2 * sig_max, 2 * sig_min, angle=np.degrees(theta),
            fill=False, color=ORANGE, lw=2.0)
)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
images = (R @ (np.array([[sig_max, 0], [0, sig_min]]) @ pts.T)).T
ax.plot(images[:, 0], images[:, 1], "o", color=ACCENT, ms=6)
ax.axhline(0, color="#c9c2b2", lw=0.8)
ax.axvline(0, color="#c9c2b2", lw=0.8)
ax.annotate(
    r"$\sigma_{\max}=2$",
    xy=(sig_max * np.cos(theta), sig_max * np.sin(theta)),
    xytext=(0.55, 1.9),
    color=GRAY,
    fontsize=11,
    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0),
)
ax.annotate(
    r"$\sigma_{\min}=0.25$",
    xy=(-sig_min * np.sin(theta), sig_min * np.cos(theta)),
    xytext=(-2.0, 1.6),
    color=GRAY,
    fontsize=11,
    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0),
)
ax.text(-2.3, 2.05, r"$\kappa=\sigma_{\max}/\sigma_{\min}=8$", color=ACCENT, fontsize=12)
ax.text(-2.3, -1.85, "flat direction: the two images nearly coincide", color=GRAY, fontsize=10)
ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-2.0, 2.4)
ax.set_aspect("equal")
ax.set_title("after A: ellipse", loc="left")
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
save(fig, "condition-number")
