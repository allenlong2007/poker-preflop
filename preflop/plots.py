"""13x13 range charts and simple line charts (matplotlib, saved as PNG)."""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .cards import HANDS

# One-hue sequential ramp: near-white = 0%, dark blue = 100%.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f6f9", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"


# Diverging ramp for ratios around 1.0: red = below, warm gray = 1.0, blue = above.
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#a3302f", "#e34948", "#f2a8a6", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"])


def range_grid(values, title, path, fmt="pct", vmin=0.0, vmax=1.0, cmap=None, note=None):
    """Draw a 13x13 hand grid. `values` has one number per class, in HANDS order.

    fmt: "pct" (action frequencies), "eq" (equities), "ratio" (e.g. realization, centred on 1).
    Missing values (NaN) are left blank.
    """
    fig, ax = plt.subplots(figsize=(8, 8.4))
    grid = [[values[r * 13 + c] for c in range(13)] for r in range(13)]
    ax.imshow(grid, cmap=cmap or (DIVERGING if fmt in ("ratio", "signed") else BLUES), vmin=vmin, vmax=vmax)
    for i, (h, v) in enumerate(zip(HANDS, values)):
        r, c = divmod(i, 13)
        if fmt == "ratio":
            dark = abs(v - 1) / max(vmax - 1, 1 - vmin) > 0.55 if v == v else False
            sub = f"{v:.2f}" if v == v else ""
        elif fmt == "signed":             # e.g. profit in bb, centred on 0
            dark = abs(v) / max(vmax, -vmin) > 0.55 if v == v else False
            sub = f"{v:+.2f}" if v == v else ""
        elif fmt == "pct":
            dark = (v - vmin) / (vmax - vmin) > 0.55
            sub = "" if v > 0.995 or v < 0.005 else f"{v:.0%}"
        else:
            dark = (v - vmin) / (vmax - vmin) > 0.55
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
    fig.text(0.125, 0.04, note or "Pairs on the diagonal, suited above it, offsuit below.",
             fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_plot(x, y, curves, title, xlabel, ylabel, path):
    """Binned data as dots plus model curves. curves = {label: (x list, y list)}."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    ax.text(0.97, 0.93, "raw equity", color=MUTED, fontsize=8.5, ha="right")
    for k, (label, (cx, cy)) in enumerate(curves.items()):
        ax.plot(cx, cy, color=["#2a78d6", "#eb6834"][k % 2], linewidth=2, label=label)
    ax.plot(x, y, "o", color=INK, markersize=4, label="simulated (binned)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    _style(ax)
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


def multi_line(series, title, xlabel, ylabel, path, legend_title="BTN open"):
    """Several lines whose labels are ordered (e.g. open sizes): one-hue light->dark ramp."""
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, (label, (x, y)) in enumerate(series.items()):
        color = ramp[round(k * (len(ramp) - 1) / max(len(series) - 1, 1))]
        ax.plot(x, y, color=color, linewidth=2, label=label)
        best = max(range(len(y)), key=lambda j: y[j])
        ax.plot([x[best]], [y[best]], marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title=legend_title, title_fontsize=9,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
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


RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def _ramp(k, n):
    return RAMP[round(k * (len(RAMP) - 1) / max(n - 1, 1))]


def _style(ax):
    ax.grid(axis="y", color=GRID)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def small_multiples(panels, title, xlabel, path, legend_title=""):
    """One panel per measure (each with its own y-axis), same ordered series in every panel.

    panels = {panel title: {series label: (x list, y list)}}
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    for ax, (ptitle, series) in zip(axes, panels.items()):
        for k, (label, (x, y)) in enumerate(series.items()):
            ax.plot(x, y, color=_ramp(k, len(series)), linewidth=2, label=label)
        ax.set_title(ptitle, loc="left", fontsize=11, color=INK)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        _style(ax)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK, title=legend_title, title_fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(df, title, xlabel, ylabel, path):
    """Generic heatmap of a DataFrame (index = rows, columns = columns); darker = larger."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    vals = df.values
    im = ax.imshow(vals, cmap=BLUES, aspect="auto")
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            dark = (v - lo) / (hi - lo + 1e-12) > 0.55
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(df.columns)), [f"{c:g}" for c in df.columns])
    ax.set_yticks(range(len(df.index)), [f"{i:g}" for i in df.index])
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8).outline.set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]


def stacked_bars(labels, parts, title, ylabel, path, colors=None, markers=None, marker_label=""):
    """100%-style stacked bars. parts = {segment name: list of values (one per label)}.

    markers: optional list of y-values drawn as a short black tick on each bar.
    """
    from matplotlib.lines import Line2D

    colors = colors or dict(zip(parts, CATEGORICAL))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for name, vals in parts.items():
        vals = np.asarray(vals, dtype=float)
        ax.bar(x, vals, 0.62, bottom=bottom, color=colors[name], label=name,
               edgecolor="white", linewidth=2)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 0.06:
                ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color=INK if colors[name] in ("#e4e3df",) else "white")
        bottom += vals
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[n]) for n in parts]
    names = list(parts)
    if markers is not None:
        for xi, m in zip(x, markers):
            ax.plot([xi - 0.36, xi + 0.36], [m, m], color=INK, linewidth=2)
        handles.append(Line2D([0], [0], color=INK, linewidth=2))
        names.append(marker_label)
    ax.legend(handles, names, frameon=False, fontsize=9, labelcolor=INK,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    _style(ax)
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
