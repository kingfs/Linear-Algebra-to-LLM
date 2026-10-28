import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(7.6, 5.6))
st.axes2d(ax)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

xg = np.linspace(-2.0, 2.0, 240)
yg = np.linspace(-1.4, 1.4, 200)
X, Y = np.meshgrid(xg, yg)
Z = 0.5 * (X**2 + 2 * Y**2)
levels = np.linspace(0.15, 4.0, 12)
ax.contour(X, Y, Z, levels=levels, colors=st.GRAY, linewidths=0.8, alpha=0.8)

xq = np.linspace(-1.8, 1.8, 9)
yq = np.linspace(-1.2, 1.2, 7)
Xq, Yq = np.meshgrid(xq, yq)
U = Xq
V = 2 * Yq
ax.quiver(Xq, Yq, U, V, color=st.ACCENT, alpha=0.55, width=0.004,
          angles="xy", scale_units="xy", scale=1.2, pivot="mid")

p = np.array([1.1, 0.7])
g = np.array([p[0], 2 * p[1]])
st.arrow(ax, p, p + 0.75 * g, st.RED, lw=2.3, zorder=6)
tan = np.array([-g[1], g[0]])
tan = tan / np.linalg.norm(tan)
ax.plot([p[0] - 0.9 * tan[0], p[0] + 0.9 * tan[0]],
        [p[1] - 0.9 * tan[1], p[1] + 0.9 * tan[1]],
        color=st.ORANGE, lw=1.8, zorder=5)
ax.scatter([p[0]], [p[1]], color=st.RED, s=30, zorder=7)

ax.text(p[0] + 0.18, p[1] + 0.78, r"$\nabla f$", color=st.RED, fontsize=12)
ax.text(p[0] - 0.72, p[1] + 0.50, "tangent", color=st.ORANGE, fontsize=9)
ax.text(-2.05, 1.38, r"$f=\frac{1}{2}(x_1^2+2x_2^2)$", fontsize=11)
ax.text(-2.05, 1.15, r"$\nabla f=(x_1,2x_2)$", fontsize=10)
ax.text(-2.05, -1.48, "arrows point uphill; contours are level sets", color=st.GRAY, fontsize=9)

ax.set_title("Gradient: uphill and perpendicular to level sets", loc="left")
st.save(fig, "gradient-field")
