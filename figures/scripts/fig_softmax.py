import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, save


def softmax(z, T):
    z = np.asarray(z) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


logits = np.array([2.0, 1.0, 0.5, -1.0])
labels = ["A", "B", "C", "D"]
temps = [0.5, 1.0, 2.0]
colors = [ACCENT, ORANGE, GREEN]
x = np.arange(len(logits))
width = 0.26

fig, ax = plt.subplots(figsize=(6.2, 4.2))
for i, (T, c) in enumerate(zip(temps, colors)):
    p = softmax(logits, T)
    ax.bar(x + (i - 1) * width, p, width, color=c, alpha=0.85, label=f"T = {T}")
    for xi, pi in zip(x + (i - 1) * width, p):
        ax.text(xi, pi + 0.012, f"{pi:.2f}", ha="center", fontsize=8, color="#52606d")
ax.set_xticks(x, labels)
ax.set_ylabel("probability")
ax.set_ylim(0, 0.95)
ax.legend(frameon=False, ncol=3, loc="upper right")
ax.set_title("Softmax temperature: low T sharpens, high T flattens")
ax.grid(True, axis="y", alpha=0.6)
save(fig, "softmax-temperature")
