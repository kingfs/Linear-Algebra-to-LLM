import numpy as np
import matplotlib.pyplot as plt
import _style as st

A = np.array([1.0, 2.0])
b = 3.0
p = A * (b / (A @ A))
q = np.array([3.0, 0.0])
theta = np.linspace(0, 2 * np.pi, 400)

fig, ax = plt.subplots(figsize=(7.8, 5.4))
st.axes2d(ax)
ax.set_xlim(-1.5, 4.0)
ax.set_ylim(-1.0, 2.3)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

xs = np.linspace(-1.4, 3.8, 200)
ys = (b - xs) / 2.0
ax.plot(xs, ys, color=st.ACCENT, lw=2.2, zorder=4)

r = np.linalg.norm(p)
ax.plot(r * np.cos(theta), r * np.sin(theta), color=st.GREEN, lw=1.4, ls=":", zorder=3)

st.arrow(ax, (0, 0), p, st.RED, lw=2.2, zorder=6)
st.arrow(ax, (0, 0), q, st.ORANGE, lw=1.8, ls="--", zorder=5)

# right-angle marker between the radius and the tangent line
u = p / np.linalg.norm(p)
d = np.array([2.0, -1.0]) / np.sqrt(5)
s = 0.13
mark = np.array([p, p - s * u, p - s * u + s * d, p + s * d])
ax.plot(mark[:, 0], mark[:, 1], color=st.GRAY, lw=1.0, zorder=5)

ax.scatter([p[0], q[0]], [p[1], q[1]], color=[st.RED, st.ORANGE], s=[55, 34], zorder=7)
ax.text(p[0] + 0.10, p[1] + 0.10, r"$p=A^{+}b$", color=st.RED, fontsize=11)
ax.text(q[0] - 0.25, q[1] - 0.30, r"$q=(3,0)$", color=st.ORANGE, fontsize=9)
ax.text(0.10, -0.28, "0", color=st.GRAY, fontsize=9)
ax.text(-1.40, 2.00, r"$A=[1\ \ 2],\ b=3$", fontsize=10)
ax.text(-1.40, 1.72, r"all solutions: $x_1+2x_2=3$", fontsize=9, color=st.GRAY)
ax.text(-1.40, 1.46, r"$\|p\|\approx1.34,\ \|q\|=3$", fontsize=9, color=st.GREEN)

ax.set_title("Minimum-norm solution: the closest point on the line", loc="left")
st.save(fig, "min-norm")
