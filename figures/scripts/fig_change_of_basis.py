import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(7.2, 6.2))
st.axes2d(ax, grid=False)
ax.set_xlim(-0.9, 2.25)
ax.set_ylim(-0.55, 2.05)
ax.set_xlabel("x")
ax.set_ylabel("y")

b1 = np.array([1.0, 0.5])
b2 = np.array([-0.5, 1.0])
x = np.array([1.2, 1.0])
c = np.array([1.36, 0.32])

# standard grid
for g in np.arange(-1.0, 3.01, 0.5):
    ax.plot([g, g], [-1, 3], color="#e6e0d4", lw=0.7, zorder=0)
    ax.plot([-1, 3], [g, g], color="#e6e0d4", lw=0.7, zorder=0)

# basis-B grid: lines along b1 and b2
for s in np.arange(-1.0, 3.01, 0.5):
    t = np.linspace(-1.5, 2.5, 2)
    line = s * b1[:, None] + t[None, :] * b2[:, None]
    ax.plot(line[0], line[1], color=st.ORANGE, lw=0.8, alpha=0.32, zorder=1)
for t in np.arange(-1.0, 3.01, 0.5):
    s = np.linspace(-1.5, 2.5, 2)
    line = s[None, :] * b1[:, None] + t * b2[:, None]
    ax.plot(line[0], line[1], color=st.GREEN, lw=0.8, alpha=0.28, zorder=1)

# component parallelogram
p1 = c[0] * b1
p2 = c[1] * b2
poly = np.array([[0, 0], p1, x, p2])
ax.fill(poly[:, 0], poly[:, 1], color=st.ACCENT, alpha=0.08, zorder=1)
ax.plot([p1[0], x[0]], [p1[1], x[1]], color=st.GRAY, lw=1.1, ls="--", zorder=2)
ax.plot([p2[0], x[0]], [p2[1], x[1]], color=st.GRAY, lw=1.1, ls="--", zorder=2)

st.arrow(ax, [0, 0], b1, st.ORANGE, lw=2.2)
st.arrow(ax, [0, 0], b2, st.GREEN, lw=2.2)
st.arrow(ax, [0, 0], x, st.ACCENT, lw=2.6)

ax.text(1.03, 0.42, r"$\mathbf{b}_1$", color=st.ORANGE, fontsize=12)
ax.text(-0.42, 0.68, r"$\mathbf{b}_2$", color=st.GREEN, fontsize=12)
ax.text(1.24, 1.08, r"$\mathbf{x}$", color=st.ACCENT, fontsize=13)
ax.text(0.63, 0.80, r"$1.36\,\mathbf{b}_1$", color=st.ORANGE, fontsize=10)
ax.text(-0.18, 0.52, r"$0.32\,\mathbf{b}_2$", color=st.GREEN, fontsize=10)
ax.text(-0.80, 1.78, r"$[\mathbf{x}]_{\mathcal{B}}=(1.36,\ 0.32)$", fontsize=10)
ax.text(-0.80, -0.48, "same vector, two sets of rulers", color=st.GRAY, fontsize=10)

ax.set_title("Change of basis: the vector stays, coordinates change", loc="left")
st.save(fig, "change-of-basis")
