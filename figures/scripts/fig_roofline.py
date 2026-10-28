import numpy as np
import matplotlib.pyplot as plt
import _style as st

fig, ax = plt.subplots(figsize=(8.4, 5.2))
ai = np.logspace(-0.6, 3.5, 500)
fp32 = np.minimum(20.0, 2.0 * ai)
bf16 = np.minimum(300.0, 2.0 * ai)

ax.loglog(ai, fp32, color=st.ACCENT, lw=2.2, label="fp32: 20 TFLOPS, 2 TB/s")
ax.loglog(ai, bf16, color=st.GREEN, lw=2.2, label="bf16: 300 TFLOPS, 2 TB/s")
ax.axvline(10, color=st.ACCENT, lw=1.0, ls=":")
ax.axvline(150, color=st.GREEN, lw=1.0, ls=":")
ax.axvline(683, color=st.ORANGE, lw=1.2, ls="--")

ax.scatter([683], [20], color=st.ACCENT, s=36, zorder=5)
ax.scatter([683], [300], color=st.GREEN, s=36, zorder=5)
ax.annotate("n=4096 GEMM\nAI=683", xy=(683, 20), xytext=(900, 3.0),
            fontsize=9, color=st.ORANGE,
            arrowprops=dict(arrowstyle="->", color=st.ORANGE, lw=1.0))
ax.text(9, 380, "ridge 10", color=st.ACCENT, fontsize=8, ha="right")
ax.text(135, 380, "ridge 150", color=st.GREEN, fontsize=8, ha="right")
ax.text(430, 380, "compute-bound", color=st.GRAY, fontsize=9)
ax.text(1.2, 0.9, "memory-bound", color=st.GRAY, fontsize=9)

ax.set_xlim(0.6, 3000)
ax.set_ylim(0.6, 600)
ax.set_xlabel("arithmetic intensity (FLOP/byte)")
ax.set_ylabel("attainable TFLOPS")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", frameon=False, fontsize=8)

ax.set_title("Roofline: bandwidth slope meets a compute ceiling", loc="left")
st.save(fig, "roofline")
