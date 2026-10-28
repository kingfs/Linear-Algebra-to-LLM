import numpy as np
import matplotlib.pyplot as plt
from _style import ACCENT, ORANGE, GREEN, GRAY, save

T = 6
rng = np.random.default_rng(7)
Q = rng.normal(size=(T, 4))
K = rng.normal(size=(T, 4))
S = Q @ K.T / np.sqrt(4)

mask = np.triu(np.ones((T, T), dtype=bool), k=1)
S_eff = np.where(mask, -np.inf, S)
P = np.exp(S_eff - S_eff.max(axis=1, keepdims=True))
P = P / P.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))

cmap_s = plt.cm.Blues.copy()
cmap_s.set_bad("#e5e7eb")
ax = axes[0]
ax.imshow(np.ma.masked_where(mask, S), cmap=cmap_s, vmin=-2.5, vmax=2.5)
for i in range(T):
    for j in range(T):
        if j > i:
            ax.text(j, i, r"$-\infty$", ha="center", va="center", color=GRAY, fontsize=9)
        else:
            ax.text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", color="#1f2933", fontsize=9)
ax.set_title("Step 1: scores + causal mask", loc="left")

cmap_p = plt.cm.Oranges.copy()
cmap_p.set_bad("#e5e7eb")
ax = axes[1]
ax.imshow(np.ma.masked_where(mask, P), cmap=cmap_p, vmin=0, vmax=1)
for i in range(T):
    for j in range(T):
        if j > i:
            ax.text(j, i, "0", ha="center", va="center", color=GRAY, fontsize=9)
        else:
            color = "white" if P[i, j] > 0.55 else "#1f2933"
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
ax.set_title("Step 2: row-wise softmax = attention", loc="left")

for ax in axes:
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(range(1, T + 1))
    ax.set_yticklabels(range(1, T + 1))
    ax.set_xlabel("key position j")
    ax.set_ylabel("query position i")
    ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
    ax.grid(which="minor", color="#fffdf8", linewidth=1.2)

fig.text(0.5, 0.01, "Each row sums to 1; the upper triangle is forbidden.", ha="center", color=GRAY, fontsize=9)
fig.subplots_adjust(wspace=0.34, bottom=0.16)
save(fig, "attention-mask")
