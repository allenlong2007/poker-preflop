"""13x13 range charts and simple line charts (matplotlib, saved as PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .cards import HANDS

# One-hue sequential ramp: near-white = 0%, dark blue = 100%.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f6f9", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"


def range_grid(values, title, path, fmt="pct", vmin=0.0, vmax=1.0):
    """Draw a 13x13 hand grid. `values` has one number per class, in HANDS order."""
    fig, ax = plt.subplots(figsize=(8, 8.4))
    grid = [[values[r * 13 + c] for c in range(13)] for r in range(13)]
    ax.imshow(grid, cmap=BLUES, vmin=vmin, vmax=vmax)
    for i, (h, v) in enumerate(zip(HANDS, values)):
        r, c = divmod(i, 13)
        dark = (v - vmin) / (vmax - vmin) > 0.55
        if fmt == "pct":
            sub = "" if v > 0.995 or v < 0.005 else f"{v:.0%}"
        else:
            sub = f"{v:.1%}"
        color = "white" if dark else INK
        ax.text(c, r - (0.12 if sub else 0), h, ha="center", va="center", fontsize=8.5,
                color=color, fontweight="bold")
        if sub:
            ax.text(c, r + 0.25, sub, ha="center", va="center", fontsize=6.5, color=color)
    ax.set_xticks([x - 0.5 for x in range(1, 13)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 13)], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    fig.text(0.125, 0.04, "Pairs on the diagonal, suited above it, offsuit below.",
             fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


ACTION_COLORS = {"fold": "#e4e3df", "call": "#2a78d6", "3-bet": "#eb6834"}


def action_grid(actions, title, path, subtitle=""):
    """13x13 grid where each cell is split left-to-right by action frequency.

    `actions` maps an action name (key of ACTION_COLORS) to a 169-length array.
    """
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(8, 8.6))
    gap = 0.04
    for i, h in enumerate(HANDS):
        r, c = divmod(i, 13)
        x = c - 0.5 + gap
        width = 1 - 2 * gap
        for name, freq in actions.items():
            w = width * freq[i]
            if w > 0.002:
                ax.add_patch(Rectangle((x, r - 0.5 + gap), w, 1 - 2 * gap,
                                       color=ACTION_COLORS[name], linewidth=0))
            x += w
        dark = actions.get("call", [0] * 169)[i] + actions.get("3-bet", [0] * 169)[i] > 0.5
        ax.text(c, r, h, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white" if dark else INK)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(12.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=24)
    if subtitle:
        ax.text(-0.5, -0.75, subtitle, fontsize=9, color=MUTED)
    ax.legend(handles=[Patch(color=v, label=k) for k, v in ACTION_COLORS.items() if k in actions],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK)
    fig.text(0.125, 0.06, "Pairs on the diagonal, suited above it, offsuit below. "
             "Each cell is split by how often the hand takes each action.",
             fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def multi_line(series, title, xlabel, ylabel, path):
    """Several lines whose labels are ordered (e.g. open sizes): one-hue light->dark ramp."""
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, (label, (x, y)) in enumerate(series.items()):
        color = ramp[round(k * (len(ramp) - 1) / max(len(series) - 1, 1))]
        ax.plot(x, y, color=color, linewidth=2, label=label)
        best = max(range(len(y)), key=lambda j: y[j])
        ax.plot([x[best]], [y[best]], marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title="BTN open", title_fontsize=9)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def line(x, y, title, xlabel, ylabel, path, highlight=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="#2a78d6", linewidth=2, marker="o", markersize=6)
    if highlight is not None:
        hx, hy = highlight
        ax.plot([hx], [hy], marker="o", markersize=9, color="#2a78d6",
                markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{hx}bb: {hy:+.3f}", (hx, hy), textcoords="offset points",
                    xytext=(8, -14), fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
