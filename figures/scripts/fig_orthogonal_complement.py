import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import _style as st


def iso(x, y, z):
    """Simple isometric view: x right, y back-left, z up."""
    return np.array([x - 0.5 * y, 0.28 * x + 0.28 * y + z])


fig, ax = plt.subplots(figsize=(7.6, 6.2))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-2.2, 2.0)
ax.set_ylim(-0.9, 2.8)

corners = np.array([[-1.2, -1.2, 0], [1.2, -1.2, 0],
                    [1.2, 1.2, 0], [-1.2, 1.2, 0]])
poly = np.array([iso(*c) for c in corners])
ax.add_patch(Polygon(poly, closed=True, facecolor=st.ACCENT, edgecolor=st.ACCENT,
                     alpha=0.12, lw=1.4, zorder=1))

for t in np.linspace(-1.2, 1.2, 5):
    p0, p1 = iso(t, -1.2, 0), iso(t, 1.2, 0)
    q0, q1 = iso(-1.2, t, 0), iso(1.2, t, 0)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=st.ACCENT, lw=0.6, alpha=0.25, zorder=2)
    ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=st.ACCENT, lw=0.6, alpha=0.25, zorder=2)

origin = iso(0, 0, 0)
normal_top = iso(0, 0, 1.95)
normal_bottom = iso(0, 0, -0.55)
ax.plot([normal_bottom[0], normal_top[0]], [normal_bottom[1], normal_top[1]],
        color=st.GRAY, lw=1.6, ls="--", zorder=3)
st.arrow(ax, normal_bottom, normal_top, st.GRAY, lw=1.6, ls="--", zorder=3)

p = np.array([1.0, 0.7, 0.0])
v = np.array([1.0, 0.7, 1.3])
e = np.array([0.0, 0.0, 1.3])

st.arrow(ax, origin, iso(*v), st.ORANGE, lw=2.4, zorder=6)
st.arrow(ax, origin, iso(*p), st.ACCENT, lw=2.2, zorder=6)
st.arrow(ax, iso(*p), iso(*v), st.GREEN, lw=2.2, ls="--", zorder=6)

# right-angle marker at p, in the plane and along the normal
u = np.array([1.0, 0.0, 0.0])
z = np.array([0.0, 0.0, 1.0])
s = 0.16
mark = np.array([p, p + s * u, p + s * u + s * z, p + s * z])
mark2 = np.array([iso(*q) for q in mark])
ax.plot(mark2[:, 0], mark2[:, 1], color=st.GRAY, lw=1.0, zorder=5)

ax.text(*iso(-1.35, -0.95, 0), r"$S$", fontsize=15, color=st.ACCENT)
ax.text(*(normal_top + np.array([0.08, 0.05])), r"$S^\perp$", fontsize=14, color=st.GRAY)
ax.text(*(iso(*v) + np.array([0.10, 0.06])), r"$v$", fontsize=14, color=st.ORANGE)
ax.text(*(iso(*p) + np.array([0.06, -0.22])), r"$p$", fontsize=13, color=st.ACCENT)
ax.text(*(iso(0.55, 0.7, 0.65)), r"$e$", fontsize=13, color=st.GREEN)

ax.text(-2.15, 2.52, r"$v=p+e$", fontsize=11)
ax.text(-2.15, 2.20, r"$p\in S,\ e\in S^\perp$", fontsize=9.5)
ax.text(-2.15, -0.78, r"$\dim S+\dim S^\perp=3$", fontsize=10, color=st.GRAY)

ax.set_title("Orthogonal complement: the perpendicular half", loc="left")
st.save(fig, "orthogonal-complement")
