import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, save

T = np.arange(1, 4097)
B, H, dk, L, bytes_per = 1, 32, 128, 32, 2
mem = 2 * B * H * T * dk * bytes_per * L / 1e9

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(T, mem, color=ACCENT, lw=2.2)
for t in [1024, 2048, 4096]:
    m = 2 * B * H * t * dk * bytes_per * L / 1e9
    ax.scatter([t], [m], color=ORANGE, s=26, zorder=4)
    ax.annotate(f"T={t}\n{m:.1f} GB", (t, m), textcoords="offset points", xytext=(-10, 8), fontsize=9, color=ORANGE)
ax.set_xlabel("sequence length T")
ax.set_ylabel("KV cache (GB)")
ax.set_title("KV cache grows linearly with T (B=1, H=32, dk=128, L=32, fp16)")
ax.grid(True, alpha=0.6)
save(fig, "kv-cache")
