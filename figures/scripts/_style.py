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
