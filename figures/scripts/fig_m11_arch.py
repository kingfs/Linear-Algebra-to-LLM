"""M11 架构扩展的六张解释图。

运行：
    uv run --with matplotlib --with numpy python figures/scripts/fig_m11_arch.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _style import ACCENT, ORANGE, GREEN, RED, GRAY, save


def fig_mixer_complexity():
    T = np.logspace(3, 5.5, 200)
    D, N, w = 4096, 16, 1024
    attn = 4 * T**2 * D
    linear = 8 * T * D**2
    ssm = 4 * T * D * N
    swa = 4 * T * w * D

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(T, attn, color=ACCENT, lw=2.2, label="full attention  O(T^2 D)")
    ax.loglog(T, swa, color=ORANGE, lw=2.0, label="SWA  O(T w D)")
    ax.loglog(T, linear, color=GREEN, lw=2.0, label="linear attention  O(T D^2)")
    ax.loglog(T, ssm, color=RED, lw=2.0, ls="--", label="SSM  O(T D N)")
    ax.axvline(2 * D, color=GRAY, lw=1.0, ls=":")
    ax.text(2 * D * 1.05, attn[0] * 0.8, "T = 2D", color=GRAY, fontsize=10)
    ax.set_xlabel("context length T")
    ax.set_ylabel("relative FLOPs per layer")
    ax.set_title("Sequence mixers: cost vs context length")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "m11-mixer-complexity")


def _box(ax, xy, text, color, width=1.9, height=0.75, fontsize=10):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.4, edgecolor=color, facecolor="white", zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="#1f2933", zorder=4)


def _arrow(ax, start, end, color=GRAY, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.6, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2,
    ))


def fig_mamba_state():
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    _box(ax, (0.9, 2.2), r"$x_t$", ACCENT)
    _box(ax, (3.0, 3.55), r"$\Delta_t, B_t, C_t$" + "\n" + r"$= f(x_t)$", ORANGE, width=2.2, height=0.95, fontsize=9)
    _box(ax, (4.9, 2.2), r"$h_t = e^{\Delta A} h_{t-1}$" + "\n" + r"$+\ \Delta B x_t$", GREEN, width=2.5, height=1.05, fontsize=9)
    _box(ax, (7.6, 2.2), r"$y_t = C h_t + D x_t$", ACCENT, width=2.2, height=0.9, fontsize=9)
    _box(ax, (4.9, 0.55), "fixed state  [D, N]", GRAY, width=2.2, height=0.6, fontsize=9)

    _arrow(ax, (1.85, 2.2), (3.65, 2.2), ACCENT)
    _arrow(ax, (4.05, 3.15), (4.05, 2.8), ORANGE)
    _arrow(ax, (6.2, 2.2), (6.5, 2.2), GREEN)
    _arrow(ax, (4.9, 1.68), (4.9, 0.9), GREEN)
    ax.add_patch(FancyArrowPatch(
        (4.0, 1.45), (5.8, 1.45), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.6, color=GREEN, connectionstyle="arc3,rad=-1.6", zorder=2,
    ))
    ax.text(4.9, 1.32, "recurrence: no KV cache", ha="center", va="top", fontsize=9, color=GREEN)
    ax.text(0.9, 1.45, "token", ha="center", fontsize=9, color=GRAY)
    ax.text(7.6, 3.45, "output", ha="center", fontsize=9, color=GRAY)
    ax.set_title("Selective SSM: write / forget / read", fontsize=13, weight="bold")
    save(fig, "m11-mamba-state")


def fig_linear_attention():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    rng = np.random.default_rng(7)
    T = 14
    scores = rng.random((T, T))
    scores[np.triu_indices(T, 1)] = np.nan
    ax = axes[0]
    im = ax.imshow(scores, cmap="YlGnBu", aspect="equal")
    ax.set_title("softmax attention\nT × T scores", fontsize=11)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    _box(ax, (1.4, 2.4), r"$k_t, v_t$", ACCENT, width=1.6, height=0.65, fontsize=10)
    _box(ax, (4.2, 2.4), r"$S_t = S_{t-1}$" + "\n" + r"$+\ \phi(k_t)v_t^\top$", GREEN, width=2.0, height=0.95, fontsize=9)
    _box(ax, (4.2, 0.8), r"$D \times D$ state", GRAY, width=1.8, height=0.55, fontsize=9)
    _arrow(ax, (2.2, 2.4), (3.2, 2.4), ACCENT)
    _arrow(ax, (4.2, 1.88), (4.2, 1.1), GREEN)
    ax.text(4.2, 3.15, "no T × T matrix", ha="center", fontsize=9, color=GREEN)
    ax.set_title("linear attention\nstate accumulation", fontsize=11)
    fig.suptitle("Two ways to mix a sequence", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "m11-linear-attention")


def fig_hybrid_layers():
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    counts = [(13, GREEN, "shortconv"), (5, ORANGE, "SWA"), (6, ACCENT, "global")]
    left = 0
    for n, color, label in counts:
        ax.barh(0, n, left=left, height=0.42, color=color, edgecolor="white", linewidth=1.2, label=f"{label} × {n}")
        ax.text(left + n / 2, 0, f"{n}", ha="center", va="center", color="white", fontsize=12, weight="bold")
        left += n
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.8, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("24 layers")
    ax.set_title("LFM3 layer budget: 13 shortconv + 5 SWA + 6 global")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    save(fig, "m11-hybrid-layers")


def fig_long_context():
    T = np.logspace(3, np.log10(262144), 200)
    L, H, dk, b = 32, 32, 128, 2
    full = 2 * L * H * dk * T * b / 2**30
    w = 4096
    swa = 2 * L * H * dk * w * b / 2**30 * np.ones_like(T)
    ssm = 2048 * 16 * 24 * b / 2**30 * np.ones_like(T)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(T, full, color=ACCENT, lw=2.2, label="full attention")
    ax.loglog(T, swa, color=ORANGE, lw=2.0, label="SWA window 4096")
    ax.loglog(T, ssm, color=RED, lw=2.0, ls="--", label="SSM state (24 layers)")
    ax.set_xlabel("context length T")
    ax.set_ylabel("KV / state memory (GiB)")
    ax.set_title("Long context: what grows with T?")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=9)
    ax.annotate("128 GiB", xy=(2.6e5, full[-1]), xytext=(1.1e5, full[-1] * 0.35),
                fontsize=9, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    save(fig, "m11-long-context")


def fig_arch_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    names = ["full\nattention", "SWA", "linear", "SSM", "hybrid"]
    mem = [128, 2.0, 0.01, 0.0015, 3.0]
    flops = [1.0, 0.25, 0.0625, 0.00012, 0.2]
    colors = [ACCENT, ORANGE, GREEN, RED, GRAY]

    ax = axes[0]
    ax.bar(names, mem, color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("KV / state at 128k (GiB)")
    ax.set_title("memory")
    ax.grid(axis="y", which="both", alpha=0.3)

    ax = axes[1]
    ax.bar(names, flops, color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("relative FLOPs at 128k")
    ax.set_title("compute")
    ax.grid(axis="y", which="both", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Architecture bill at 128k: memory vs compute", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "m11-arch-tradeoff")


if __name__ == "__main__":
    fig_mixer_complexity()
    fig_mamba_state()
    fig_linear_attention()
    fig_hybrid_layers()
    fig_long_context()
    fig_arch_tradeoff()
