"""shared plot style: palette, matplotlib defaults, and the operator order.

categorical slots are assigned in fixed order and never cycled, so an operator
keeps its colour in every figure. three slots sit under 3:1 contrast on this
surface, so every figure ships with a legend and a markdown table beside it.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PLOTS = REPO / "results" / "plots"
TABLES = REPO / "results" / "tables"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

# fixed order, eight slots: quantize folds into other rather than take a ninth hue
OPS = ["conv2d", "depthwise_conv", "pointwise_conv", "batch_norm", "relu", "pool", "dense", "other"]
OP_COLOR = dict(zip(OPS, CATEGORICAL))


def op_bucket(op):
    return op if op in OP_COLOR else "other"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 9.5, "axes.titlesize": 11, "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def sequential(fraction):
    # step of the blue ramp for a value in [0, 1]
    idx = min(len(SEQUENTIAL) - 1, max(0, round(fraction * (len(SEQUENTIAL) - 1))))
    return SEQUENTIAL[idx]


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def write_table(name, text):
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.md"
    path.write_text(text)
    print(f"wrote {path}")
    return path


def markdown(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]

    def line(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    return "\n".join([line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)]) + "\n"
