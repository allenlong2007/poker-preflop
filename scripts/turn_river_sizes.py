"""Experiment 30: finer turn and river bet sizes for the match bot.

Experiment 26's postflop policies have two or three sizes on the turn (1/2, pot) and river (1/2, pot, 1.5x),
so the bot had to round every other bet to one of them. The policies are retrained with

    turn   bets 1/3, 1/2, 3/4, pot, 1.5x       leads 1/3, 1/2, 3/4, pot
    river  bets 1/3, 1/2, 3/4, pot, 1.5x, 2x   leads 1/3, 1/2, 3/4, pot, 1.5x

(`OPTS_FINE`; `match_strategy.py postflop-fine`), and the engine blends the two neighbouring trained sizes
when facing anything in between (pseudo-harmonic mapping, `bot.engine.size_weights`).

Metrics:
  usage     which sizes each player uses on the turn and river, by hand group, line and board
  defence   how often each player folds to each size, vs the break-even bluffing frequency x / (1 + x)
  value     BTN result per pot, old vs fine sizes
  matches   duplicate matches: fine vs old policies, and both vs an opponent with off-grid sizes
            (rounding to the nearest size vs blending)

Usage:
    python scripts/turn_river_sizes.py report      (usage, defence, value; seconds)
    python scripts/turn_river_sizes.py matches [deals]   (default 100000 deals per match)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bb_short_stack_study import cell_mask, node_freq
from bot.engine import POLICY, POLICY_FINE, POTS, Engine, _opts_from_json
from bot.selfplay import EngineAgent, match
from preflop import fullgame as fg
from preflop.plots import GRID, INK, MUTED, heatmap
from preflop.rivergame import TEXTURES as RIVER_TEX
from preflop.turngame import TEXTURES as TURN_TEX

OUT = ROOT / "output" / "turn_river_sizes"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = {"two pair+": [0, 1, 2], "overpair / top pair": [3, 4, 5], "middle / weak pair": [6, 7],
          "draws (missed on the river)": [8, 9], "overcards / air": [10, 11]}
LINES = {"no bet last street": [0, 1, 2], "BTN bet last street": [3, 4, 5], "BB bet last street": [6, 7, 8]}
STREET_NAMES = {1: "turn", 2: "river"}
NODE_NAMES = {"btn_vs_check": "BTN after the BB checks", "bb_first": "BB first (lead)"}


def nm(x):
    return {1 / 3: "1/3", 1 / 2: "1/2", 2 / 3: "2/3", 3 / 4: "3/4", 1.0: "pot", 1.5: "1.5x", 2.0: "2x"}.get(x, f"{x:g}")


def load(folder, pot):
    d = np.load(folder / f"{pot}.npz")
    if "opts_json" in d.files:
        opts = _opts_from_json(str(d["opts_json"]))
    else:
        from open_size_study import OPTS_MATCH
        opts = OPTS_MATCH
    return d, opts


def labels(street, kind, opts):
    leads, bets = fg.street_sizes(street, opts)
    return {"bb_first": ["check"] + [f"lead {nm(x)}" for x in leads],
            "btn_vs_check": ["check"] + [f"bet {nm(x)}" for x in bets]}[kind]


def usage(d, opts, street, kind):
    """Rows: all hands, hand groups, lines, board textures. Columns: check + each size."""
    names = labels(street, kind, opts)
    shp = d[f"reach_{street}_{kind}"].shape
    rows = {"all hands": node_freq(d, street, kind)}
    for g, b in GROUPS.items():
        rows[g] = node_freq(d, street, kind, cell_mask(shp, buckets=b))
    for g, ls in LINES.items():
        rows[g] = node_freq(d, street, kind, cell_mask(shp, lines=ls))
    tex_names = [f"{a}, {b}" for a, b in TURN_TEX] if street == 1 else RIVER_TEX
    for t, tn in enumerate(tex_names):
        m = np.zeros(shp)
        m[t] = 1
        if (d[f"reach_{street}_{kind}"] * m).sum() > 0:          # skip impossible boards
            rows[f"board: {tn}"] = node_freq(d, street, kind, m)
    df = pd.DataFrame(rows, index=names).T
    return df.div(df.sum(axis=1), axis=0)


def defence(d, opts, street, kind):
    """Fold / call / raise frequency by size faced (all hands)."""
    leads, bets = fg.street_sizes(street, opts)
    sizes = bets if kind == "bb_vs_bet" else leads
    shp = d[f"reach_{street}_{kind}"].shape
    rows = []
    for k, x in enumerate(sizes):
        f = node_freq(d, street, kind, cell_mask(shp, size=k))
        f = f / max(f.sum(), 1e-12)
        rows.append({"size": x, "fold": f[0], "call": f[1], "raise": f[2:].sum(),
                     "break-even bluff": x / (1 + x)})
    return pd.DataFrame(rows)


def defence_chart(curves, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"turn, BB vs BTN bet": "#2a78d6", "river, BB vs BTN bet": "#0d366b",
              "turn, BTN vs BB lead": "#f6a07a", "river, BTN vs BB lead": "#c4501f"}
    xs = np.linspace(0.3, 2.05, 50)
    ax.plot(xs, xs / (1 + xs), color=MUTED, linestyle="--", linewidth=1.5, label="break-even x / (1 + x)")
    for name, df in curves.items():
        ax.plot(df["size"], df["fold"], marker="o", color=colors[name], linewidth=2, label=name)
    ax.set_title("How often each player folds, by bet size (fine-size policies, single-raised pot)",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel("bet size (fraction of the pot)", color=MUTED)
    ax.set_ylabel("fold frequency", color=MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def md(df):
    cols = list(df.columns)
    out = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        out.append(f"| {idx} | " + " | ".join("–" if pd.isna(r[c]) else f"{r[c]:.0%}" for c in cols) + " |")
    return "\n".join(out)


def report():
    pd.set_option("display.width", 250)
    summary = []
    for pot in POTS:
        if not (POLICY_FINE / f"{pot}.npz").exists():
            continue
        old, _ = load(POLICY, pot)
        new, _ = load(POLICY_FINE, pot)
        s = float(new["pot_to"])
        summary.append({"pot": pot, "pot_bb": 2 * s, "BTN net, old sizes": float(old["value_btn"]) - s,
                        "BTN net, fine sizes": float(new["value_btn"]) - s})
    val = pd.DataFrame(summary)
    val["change (bb per pot)"] = val["BTN net, fine sizes"] - val["BTN net, old sizes"]
    val.round(4).to_csv(OUT / "value.csv", index=False)
    print(val.round(4).to_string(index=False))

    new, o_new = load(POLICY_FINE, "srp")
    old, o_old = load(POLICY, "srp")
    tables = []
    for street in (1, 2):
        for kind in ("btn_vs_check", "bb_first"):
            u = usage(new, o_new, street, kind)
            u.round(3).to_csv(OUT / f"usage_{STREET_NAMES[street]}_{kind}.csv")
            main = u.loc[["all hands", *GROUPS, *LINES]]
            heatmap(main, f"{STREET_NAMES[street].capitalize()}: {NODE_NAMES[kind]} (single-raised pot, fine sizes)",
                    "", "", OUT / f"usage_{STREET_NAMES[street]}_{kind}.png", fmt="pct", figsize=(8.5, 4.2))
            u_old = usage(old, o_old, street, kind).loc[["all hands"]]
            u_old.index = ["all hands, old sizes"]
            both = pd.concat([u_old, u.loc[["all hands"]].rename(index={"all hands": "all hands, fine sizes"})])
            tables.append((street, kind, both[list(u.columns)]))
            print(f"\n== {STREET_NAMES[street]} {kind}")
            print(u.round(2).to_string())
    curves = {}
    for street in (1, 2):
        for kind, who in (("bb_vs_bet", "BB vs BTN bet"), ("btn_vs_lead", "BTN vs BB lead")):
            df = defence(new, o_new, street, kind)
            df.round(3).to_csv(OUT / f"defence_{STREET_NAMES[street]}_{kind}.csv", index=False)
            curves[f"{STREET_NAMES[street]}, {who}"] = df
            print(f"\n== defence {STREET_NAMES[street]} {kind}")
            print(df.round(3).to_string(index=False))
    defence_chart(curves, OUT / "defence.png")
    with open(OUT / "old_vs_fine.md", "w") as f:
        for street, kind, t in tables:
            f.write(f"### {STREET_NAMES[street]}: {NODE_NAMES[kind]}\n\n{md(t)}\n\n")


class OddSizer(EngineAgent):
    """An engine whose postflop bets are scaled by `factor` (e.g. 0.8: its 1/2-pot bet becomes 40%)."""

    def __init__(self, engine, factor, name):
        super().__init__(engine, name)
        self.factor = factor

    def act(self, seat, hole, board, history, facing):
        kind, to = super().act(seat, hole, board, history, facing)
        if kind == "bet" and len(history) > 1:
            to = max(1.0, to * self.factor)
        return kind, to


def matches(deals):
    rows = []

    def run(a, b, label):
        t0 = time.time()
        r = match(a, b, deals, seed=7)
        per100, se = r.mean() / 2 * 100, r.std(ddof=1) / np.sqrt(deals) / 2 * 100
        rows.append({"match": label, "bb/100": per100, "se": se, "hands": 2 * deals})
        print(f"{label}: {per100:+.2f} ± {se:.2f} bb/100 ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).round(3).to_csv(OUT / "matches.csv", index=False)

    fine = lambda **k: Engine(policy_dir=POLICY_FINE, seed=1, **k)
    old = lambda **k: Engine(policy_dir=POLICY, seed=2, **k)
    run(EngineAgent(fine()), EngineAgent(old()), "fine sizes vs old sizes")
    for factor, desc in ((0.8, "bets 0.8x its sizes (40%, 80% ...)"), (1.3, "bets 1.3x its sizes (65%, 130% ...)")):
        for name, eng in (("fine, blend", fine()), ("fine, nearest", fine(translate=False)),
                          ("old, blend", old()), ("old, nearest", old(translate=False))):
            opp = OddSizer(Engine(policy_dir=POLICY, seed=3), factor, desc)
            run(EngineAgent(eng), opp, f"{name} vs old bot that {desc}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "matches":
        matches(int(sys.argv[2]) if len(sys.argv) > 2 else 100_000)
    else:
        report()


if __name__ == "__main__":
    main()
