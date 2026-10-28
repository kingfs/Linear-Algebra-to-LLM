import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4))
pts = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 2.0]])

for ax in axes:
    st.axes2d(ax)
    ax.set_xlim(-1.6, 4.6)
    ax.set_ylim(-1.7, 5.6)
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    ax.scatter(pts[:, 0], pts[:, 1], color=st.GRAY, s=34, zorder=5)
    for i, p in enumerate(pts):
        ax.text(p[0] + 0.10, p[1] + 0.10, rf"$x_{i+1}$", color=st.GRAY, fontsize=9)

ax = axes[0]
ln = pts.copy()
for i, p in enumerate(pts):
    mu = p.mean()
    sd = p.std()
    ln[i] = (p - mu) / sd
    st.arrow(ax, p, ln[i], "#c9c2b2", lw=1.0, ls="--", zorder=2)
ax.axline((0, 0), slope=-1, color=st.GREEN, lw=1.4, ls=":")
ax.scatter(ln[:, 0], ln[:, 1], color=st.GREEN, s=42, zorder=6)
ax.text(-1.45, 4.95, r"LayerNorm: $\mu=0$, $\sigma=1$", fontsize=10)
ax.text(-1.45, 4.62, r"outputs lie on $x_1+x_2=0$", fontsize=9, color=st.GREEN)
ax.text(-1.45, -1.40, "D=2 toy case: every direction collapses to two signs", fontsize=8.5)
ax.set_title("LayerNorm: remove the mean, then scale", loc="left", fontsize=11)

ax = axes[1]
rn = pts.copy()
for i, p in enumerate(pts):
    rn[i] = np.sqrt(2) * p / np.linalg.norm(p)
    st.arrow(ax, p, rn[i], "#c9c2b2", lw=1.0, ls="--", zorder=2)
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.sqrt(2) * np.cos(theta), np.sqrt(2) * np.sin(theta),
        color=st.RED, lw=1.3, ls=":")
ax.scatter(rn[:, 0], rn[:, 1], color=st.RED, s=42, zorder=6)
ax.text(-1.45, 4.95, r"RMSNorm: rescale only", fontsize=10)
ax.text(-1.45, 4.62, r"outputs keep their direction", fontsize=9, color=st.RED)
ax.text(-1.45, -1.40, r"all outputs land on the circle $\|x\|=\sqrt{2}$", fontsize=8.5)
ax.set_title("RMSNorm: keep the center, change the volume", loc="left", fontsize=11)

fig.suptitle("Normalization geometry: LayerNorm projects, RMSNorm rescales", y=1.02)
fig.tight_layout()
st.save(fig, "layernorm-rmsnorm")
