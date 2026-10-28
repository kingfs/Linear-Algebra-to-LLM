import numpy as np
import matplotlib.pyplot as plt
import _style as st

rng = np.random.default_rng(13)
n = 600
A = rng.normal(0, 1 / np.sqrt(n), size=(n, n))
W = (A + A.T) / np.sqrt(2)
eig = np.linalg.eigvalsh(W)

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.hist(eig, bins=64, density=True, color=st.ACCENT, alpha=0.35,
        edgecolor=st.ACCENT, linewidth=0.5, label="eigenvalues")
x = np.linspace(-2, 2, 400)
semi = np.sqrt(np.maximum(0, 4 - x**2)) / (2 * np.pi)
ax.plot(x, semi, color=st.RED, lw=2.2, label="Wigner semicircle")
ax.axvline(2, color=st.GREEN, lw=1.2, ls="--")
ax.axvline(-2, color=st.GREEN, lw=1.2, ls="--")
ax.annotate(r"$\lambda_{\max}\approx2$", xy=(2, 0.18), xytext=(2.18, 0.26),
            fontsize=10, color=st.GREEN,
            arrowprops=dict(arrowstyle="->", color=st.GREEN, lw=1.0))
ax.set_xlim(-2.35, 2.35)
ax.set_ylim(0, 0.34)
ax.set_xlabel("eigenvalue")
ax.set_ylabel("density")
ax.grid(True, alpha=0.35)
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.text(-2.25, 0.31, r"$n=600$, entries $\mathcal{N}(0,1/n)$", fontsize=9, color=st.GRAY)

ax.set_title("Wigner semicircle: random weights have a bounded spectrum", loc="left")
st.save(fig, "random-matrix")
