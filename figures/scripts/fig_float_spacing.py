import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(8.2, 4.8))
k = np.arange(-4, 11)
ulp = 2.0 ** (k - 23)
ax.step(k, ulp, where="post", color=st.ACCENT, lw=2.2)
ax.scatter(k, ulp, color=st.ACCENT, s=18, zorder=3)
ax.set_yscale("log")
ax.set_xlim(-4.5, 10.5)
ax.set_ylim(1e-11, 1e-3)
ax.set_xticks(np.arange(-4, 11, 2))
ax.set_xticklabels([rf"$2^{{{int(v)}}}$" for v in np.arange(-4, 11, 2)])
ax.set_xlabel("value lies in the octave [2^k, 2^(k+1))")
ax.set_ylabel("spacing between adjacent floats")
ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4])
ax.set_yticklabels([r"$10^{-10}$", r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$"])
ax.minorticks_off()
ax.grid(True, which="major", alpha=0.35)

ax.annotate(r"$2^{-23}\approx1.19\times10^{-7}$", xy=(0, 2**-23),
            xytext=(0.7, 3e-8), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=st.GRAY, lw=1.0))
ax.annotate(r"$2^{-13}\approx1.22\times10^{-4}$", xy=(10, 2**-13),
            xytext=(4.2, 2e-5), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=st.GRAY, lw=1.0))
ax.text(-4.2, 4e-4, "fp32: 23 mantissa bits; spacing doubles each octave",
        color=st.GRAY, fontsize=9)

ax.set_title("Floating-point spacing grows with magnitude", loc="left")
st.save(fig, "float-spacing")
