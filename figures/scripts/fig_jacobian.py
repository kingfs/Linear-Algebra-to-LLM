import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))

# input grid in (r, theta)
ax = axes[0]
ax.set_xlim(0.3, 2.15)
ax.set_ylim(-0.08, 1.65)
ax.set_xlabel("r")
ax.set_ylabel(r"$\theta$")
ax.grid(True, alpha=0.45)
r_vals = np.arange(0.4, 2.01, 0.4)
t_vals = np.arange(0.0, 1.61, 0.2)
for r in r_vals:
    ax.axvline(r, color="#e6e0d4", lw=0.8)
for t in t_vals:
    ax.axhline(t, color="#e6e0d4", lw=0.8)
ax.add_patch(plt.Rectangle((1.2, 0.6), 0.4, 0.3, facecolor=st.ORANGE,
                           alpha=0.30, edgecolor=st.ORANGE, lw=1.5))
ax.text(1.28, 1.30, r"$\Delta r=0.4$", color=st.ORANGE, fontsize=9)
ax.text(0.42, 0.70, r"$\Delta\theta=0.3$", color=st.ORANGE, fontsize=9)
ax.set_title("Input grid (r, theta)", loc="left", fontsize=11)

# image under polar map
ax = axes[1]
st.axes2d(ax)
ax.set_xlim(-2.15, 2.15)
ax.set_ylim(-2.15, 2.15)
ax.set_xlabel("x")
ax.set_ylabel("y")
theta = np.linspace(0, np.pi / 2, 240)
for r in r_vals:
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=st.ACCENT, lw=0.9, alpha=0.55)
for t in t_vals:
    ax.plot([0.4 * np.cos(t), 2.0 * np.cos(t)],
            [0.4 * np.sin(t), 2.0 * np.sin(t)], color=st.GREEN, lw=0.9, alpha=0.45)

r0, r1 = 1.2, 1.6
t0, t1 = 0.6, 0.9
corners = np.array([[r0, t0], [r1, t0], [r1, t1], [r0, t1]])
img = np.array([[r * np.cos(t), r * np.sin(t)] for r, t in corners])
ax.add_patch(plt.Polygon(img, closed=True, facecolor=st.ORANGE,
                         alpha=0.35, edgecolor=st.ORANGE, lw=1.8))
ax.text(0.15, 1.92, r"$\det J=r$", fontsize=11)
ax.text(-2.0, -1.95, "area scales by r at that radius", color=st.GRAY, fontsize=9)
ax.set_title("Image: F(r, theta) = (r cos theta, r sin theta)", loc="left", fontsize=10)

fig.suptitle("Jacobian determinant: the local area exchange rate", y=1.02)
fig.tight_layout()
st.save(fig, "jacobian-polar")
