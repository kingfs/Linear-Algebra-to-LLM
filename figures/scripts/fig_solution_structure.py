import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(7.0, 5.6))
st.axes2d(ax)
ax.set_xlim(-0.6, 4.4)
ax.set_ylim(-1.6, 3.6)
ax.set_xlabel("x")
ax.set_ylabel("y")

xx = np.linspace(-0.6, 4.4, 2)
ax.plot(xx, 2 - xx, color=st.ACCENT, lw=2.2, label="x + y = 2")

p = np.array([1.0, 1.0])
q = np.array([2.0, 0.0])
n = np.array([1.0, -1.0]) / np.sqrt(2)

ax.scatter([p[0], q[0]], [p[1], q[1]], color=[st.RED, st.GRAY], s=[50, 36], zorder=6)
ax.annotate("", xy=p + 1.35 * n, xytext=p - 1.35 * n,
            arrowprops=dict(arrowstyle="<->", color=st.GREEN, lw=1.8))
ax.annotate(r"$N(A)$", xy=p + 1.45 * n, xytext=p + 1.55 * n,
            fontsize=11, color=st.GREEN)
ax.annotate(r"$\mathbf{x}_p=(1,1)$", xy=p, xytext=(0.20, 1.30),
            fontsize=10, color=st.RED)
ax.annotate(r"$\mathbf{x}_q=(2,0)$", xy=q, xytext=(2.10, 0.42),
            fontsize=10, color=st.GRAY)
ax.plot([0, p[0]], [0, p[1]], color=st.GRAY, lw=1.0, ls="--")

ax.text(-0.45, 3.10, r"$\mathbf{x}=\mathbf{x}_p+t\,\mathbf{n}$", fontsize=12)
ax.text(-0.45, 2.78, "particular solution + any nullspace vector", color=st.GRAY, fontsize=9)
ax.text(-0.45, -1.18, "The whole solution set is this line.", fontsize=9)
ax.text(-0.45, -1.43, "p is one convenient point on it.", fontsize=9)

ax.set_title("General solution = particular solution + nullspace", loc="left")
st.save(fig, "solution-structure")
