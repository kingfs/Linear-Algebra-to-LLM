import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#fffdf8",
    "axes.facecolor": "#fffdf8",
    "axes.edgecolor": "#c9c2b2",
    "axes.labelcolor": "#1f2933",
    "xtick.color": "#52606d",
    "ytick.color": "#52606d",
    "text.color": "#1f2933",
    "grid.color": "#e6e0d4",
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "svg.fonttype": "path",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

ACCENT = "#4f46e5"
ORANGE = "#d97706"
GREEN = "#0f766e"
RED = "#b91c1c"
GRAY = "#7b8794"


def save(fig, name):
    fig.savefig(f"figures/out/{name}.svg", format="svg")
    plt.close(fig)
    print("wrote", name)


def axes2d(ax, grid=True):
    """等比例坐标轴 + 过原点的十字线，供二维几何图复用。"""
    ax.set_aspect("equal")
    ax.axhline(0, color="#c9c2b2", lw=0.8, zorder=0)
    ax.axvline(0, color="#c9c2b2", lw=0.8, zorder=0)
    if grid:
        ax.grid(True, alpha=0.5, zorder=0)


def arrow(ax, start, end, color, lw=2.2, ls="-", zorder=3):
    """从 start 指向 end 的实心箭头；坐标一律用数据坐标。"""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )
