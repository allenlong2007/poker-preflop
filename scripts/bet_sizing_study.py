"""Experiment 13: multiple bet sizes on every street.

Flop c-bet 1/3 or 3/4 pot; turn bet / lead 1/2 or 1x pot; river bet / lead 1/2, 1x
or 1.5x pot (overbet). Raises are 3x the bet. Learned street by street with
preflop/sizedgame.py on the same line as experiments 8-12 (2.25bb open; BB
checks the flop and calls a c-bet; BB check-calls a turn bet).

Usage:  python scripts/bet_sizing_study.py           (~15 min on an idle multi-core machine)
        python scripts/bet_sizing_study.py single    (one size per street: a check against
                                                      experiments 8, 11 and 12)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.flopgame import TEXTURES as FLOP_TEXTURES
from preflop.plots import grouped_bars, heatmap, stacked_bars
from preflop.rivergame import BUCKETS as RIVER_BUCKETS, TEXTURES as RIVER_TEXTURES
from preflop.sizedgame import BUCKETS, COLUMNS, SIZES, learn_street, n_lines
from preflop.turngame import TEXTURES as TURN_TEXTURES

MODE = sys.argv[1] if len(sys.argv) > 1 else "multi"
SINGLE = {"flop": (1 / 3,), "turn": (2 / 3,), "river": (2 / 3,)}
SIZE_SET = SINGLE if MODE == "single" else SIZES
OPEN = 2.25
OUT = ROOT / "output" / ("bet_sizing" if MODE == "multi" else "bet_sizing_single")
OUT.mkdir(parents=True, exist_ok=True)
POLICIES = ROOT / "output" / "cbet_frequency" / "flop_policies.npz"
GRAY, BLUE, ORANGE = "#e4e3df", "#2a78d6", "#eb6834"
SIZE_COLORS = ["#86b6ef", "#2a78d6", "#104281"]          # light -> dark = small -> big
RESP = {"fold": GRAY, "call": BLUE, "check-raise": ORANGE}


def size_name(f):
    return {1 / 3: "1/3 pot", 3 / 4: "3/4 pot", 1 / 2: "1/2 pot", 1.0: "pot", 3 / 2: "1.5x pot",
            2 / 3: "2/3 pot"}.get(f, f"{f:g} pot")


def wavg(x, w):
    w = np.asarray(w, float)
    return (np.asarray(x) * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else np.full(np.asarray(x).shape[1], np.nan)


def learn_all(f, rules):
    out = {}
    prior = {}
    for street in ("flop", "turn", "river"):
        cache = OUT / f"{street}.npz"                # local cache (gitignored); delete to re-learn
        if cache.exists():
            d = np.load(cache)
            pol = [d[f"p{m}"] for m in range(6)]
            hands = d["hands"]
        else:
            print(f"Learning the {street}:")
            res = learn_street(street, f["btn_range"], f["bb_range"], rules, prior=prior, sizes=SIZE_SET, s=OPEN,
                               rounds=50 if street != "river" else 60,
                               hands_per_round=60_000 if street != "river" else 150_000,
                               n=600_000 if street != "river" else 1_500_000)
            pol, hands = res["policy"], res["hands"]
            np.savez_compressed(cache, hands=hands, **{f"p{m}": pol[m] for m in range(6)})
        prior[street] = pol
        out[street] = (pol, pd.DataFrame(hands, columns=COLUMNS))
    return out


def street_tables(street, pol, h):
    """Per-hand policy lookups for one street, plus weights for each decision point."""
    T, tb, bk = h.texture.astype(int), h.btn_bucket.astype(int), h.bb_bucket.astype(int)
    p0, p1, p2 = pol[0][T, bk], pol[1][T, tb], pol[2][T, bk]
    w = h.weight.to_numpy()
    nl = n_lines(street, SIZE_SET)
    return {"p0": p0, "p1": p1, "p2": p2, "w": w, "w_check": w * p0[:, 0], "tex": (T // nl).to_numpy(),
            "line": (T % nl).to_numpy(), "tb": tb.to_numpy(), "bk": bk.to_numpy(), "r": h.btn_wins.to_numpy()}


def by_group(values, weight, groups, order):
    rows = {}
    for g in order:
        m = groups == g
        if weight[m].sum() > 0:
            rows[g] = wavg(values[m], weight[m])
    return rows


def report_street(street, pol, h, buckets, groupings):
    """groupings: list of (name, function texture-index array -> label array, label order) for board charts."""
    sizes = SIZE_SET[street]
    K = len(sizes)
    d = street_tables(street, pol, h)
    group_name, group_fn, group_order = groupings[0]
    d["group"] = group_fn(d["tex"])
    names = [size_name(x) for x in sizes]
    btn_cols = ["check"] + [f"bet {n}" for n in names]
    rows = []

    # 1. BTN when checked to: check or which size, by hand and by board group.
    by_hand = by_group(d["p1"], d["w_check"], np.array([buckets[b] for b in d["tb"]]), buckets)
    by_board = by_group(d["p1"], d["w_check"], d["group"], group_order)
    for k, v in by_hand.items():
        rows.append({"street": street, "table": "BTN when checked to, by hand", "row": k, **dict(zip(btn_cols, v))})
    for k, v in by_board.items():
        rows.append({"street": street, "table": f"BTN when checked to, by {group_name}", "row": k,
                     **dict(zip(btn_cols, v))})
    colors = {"check": GRAY, **{c: SIZE_COLORS[q + (3 - K)] for q, c in enumerate(btn_cols[1:])}}
    bh = pd.DataFrame(by_hand, index=btn_cols).T
    stacked_bars(list(bh.index), {c: bh[c] for c in btn_cols}, f"{street.title()}: BTN's bet size when checked to, "
                 "by hand", "Share of spots", OUT / f"{street}_btn_sizes_by_hand.png", colors=colors)
    for gname, gfn, gorder in groupings:
        bb_ = pd.DataFrame(by_group(d["p1"], d["w_check"], gfn(d["tex"]), gorder), index=btn_cols).T
        for k, v in bb_.iterrows():
            if gname != group_name:
                rows.append({"street": street, "table": f"BTN when checked to, by {gname}", "row": k,
                             **dict(zip(btn_cols, v))})
        stacked_bars(list(bb_.index), {c: bb_[c] for c in btn_cols}, f"{street.title()}: BTN's bet size when "
                     f"checked to, by {gname}", "Share of spots",
                     OUT / f"{street}_btn_sizes_by_{gname.replace(' ', '_')}.png", colors=colors)

    # 2. BB facing each size: fold / call / check-raise, vs MDF; bluff-catching results.
    resp_rows, mdf_fold_limit, hand_grid_fold, hand_grid_xr = {}, [], {}, {}
    for k, name in enumerate(names):
        wk = d["w_check"] * d["p1"][:, 1 + k]
        resp = wavg(d["p2"][:, k, :], wk)
        resp_rows[f"vs {name}"] = resp
        mdf = 1 / (1 + sizes[k])
        mdf_fold_limit.append(1 - mdf)
        calls = wk * d["p2"][:, k, 1]
        bb_wins_calling = float(((1 - d["r"]) * calls).sum() / calls.sum()) if street == "river" else np.nan
        btn_bluffs = float(((d["r"] == 0) * wk).sum() / wk.sum()) if street == "river" else np.nan
        rows.append({"street": street, "table": "BB vs a bet", "row": f"vs {name}", "fold": resp[0], "call": resp[1],
                     "check-raise": resp[2], "mdf_defend": mdf, "bb_defends": 1 - resp[0],
                     "break_even_call": sizes[k] / (1 + 2 * sizes[k]), "bb_wins_when_calling": bb_wins_calling,
                     "btn_bet_loses_showdown": btn_bluffs,
                     "btn_uses_this_size": wk.sum() / d["w_check"].sum()})     # of all checked-to spots
        hb = by_group(d["p2"][:, k, :], wk, np.array([buckets[b] for b in d["bk"]]), buckets)
        hand_grid_fold[f"vs {name}"] = {b: v[0] for b, v in hb.items()}
        hand_grid_xr[f"vs {name}"] = {b: v[2] for b, v in hb.items()}
    rr = pd.DataFrame(resp_rows, index=list(RESP)).T
    stacked_bars(list(rr.index), {c: rr[c] for c in RESP}, f"{street.title()}: BB response by BTN bet size",
                 "Share of bets faced", OUT / f"{street}_bb_vs_size.png", colors=RESP,
                 markers=mdf_fold_limit, marker_label="fold limit (MDF)")
    for title, gridd, fname in (("folds", hand_grid_fold, "fold"), ("check-raises", hand_grid_xr, "xr")):
        g = pd.DataFrame(gridd).T.reindex(columns=[b for b in buckets if b in pd.DataFrame(gridd).T.columns])
        heatmap(g, f"{street.title()}: BB {title} by BTN bet size x hand", "BB hand", "BTN bet size",
                OUT / f"{street}_bb_{fname}_by_size_hand.png", fmt="pct", figsize=(9, 3.2))

    # 3. BB first to act (turn and river only): check or lead, by hand and by board.
    if street != "flop":
        bb_cols = ["check"] + [f"lead {n}" for n in names]
        lh = by_group(d["p0"], d["w"], np.array([buckets[b] for b in d["bk"]]), buckets)
        lb = by_group(d["p0"], d["w"], d["group"], group_order)
        for k, v in lh.items():
            rows.append({"street": street, "table": "BB first to act, by hand", "row": k, **dict(zip(bb_cols, v))})
        for k, v in lb.items():
            rows.append({"street": street, "table": f"BB first to act, by {group_name}", "row": k,
                         **dict(zip(bb_cols, v))})
        lcolors = {"check": GRAY, **{c: ["#f6b89f", "#eb6834", "#a3401b"][q + (3 - K)] for q, c in enumerate(bb_cols[1:])}}
        lt = pd.DataFrame(lb, index=bb_cols).T
        stacked_bars(list(lt.index), {c: lt[c] for c in bb_cols}, f"{street.title()}: BB checks or leads (and how big), "
                     f"by {group_name}", "Share of spots",
                     OUT / f"{street}_bb_first_by_{group_name.replace(' ', '_')}.png", colors=lcolors)
        lt = pd.DataFrame(lh, index=bb_cols).T
        stacked_bars(list(lt.index), {c: lt[c] for c in bb_cols}, f"{street.title()}: BB checks or leads (and how big), "
                     "by hand", "Share of spots", OUT / f"{street}_bb_first_by_hand.png", colors=lcolors)
    return pd.DataFrame(rows)


def main():
    f = np.load(POLICIES)
    rules = {"table": f["table"], "bluff_probs": f["bluff_probs"]}
    res = learn_all(f, rules)
    favored = f["favored"]

    def flop_favor(tex):
        return np.array(["favors BTN" if favored[t] else "favors BB" for t in tex])

    def flop_high(tex):
        return np.array([FLOP_TEXTURES[t][0] for t in tex])

    reports = [
        report_street("flop", *res["flop"], BUCKETS,
                      [("board", flop_favor, ["favors BTN", "favors BB"]),
                       ("high card", flop_high, ["A-high", "K-high", "Q/J-high", "T-high or lower"])]),
        report_street("turn", *res["turn"], BUCKETS,
                      [("turn card", lambda tex: np.array([TURN_TEXTURES[t][0] for t in tex]),
                        [t for t, _ in TURN_TEXTURES[::2]])]),
        report_street("river", *res["river"], RIVER_BUCKETS,
                      [("river card", lambda tex: np.array([RIVER_TEXTURES[t] for t in tex]), RIVER_TEXTURES)]),
    ]
    allrep = pd.concat(reports)
    allrep.round(4).to_csv(OUT / "summary.csv", index=False)

    river = allrep[(allrep.street == "river") & (allrep.table == "BB vs a bet")]
    if len(river) > 1:
        grouped_bars(list(river.row), {"BB wins when it calls": list(river.bb_wins_when_calling),
                                       "break-even (pot odds)": list(river.break_even_call)},
                     "River: how often the BB's calls win, by BTN bet size", "Win rate", OUT / "river_bluff_catching.png")
    pd.set_option("display.width", 200)
    for (street, table), g in allrep.groupby(["street", "table"], sort=False):
        print(f"\n== {street}: {table}")
        print(g.drop(columns=["street", "table"]).dropna(axis=1, how="all").round(3).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
