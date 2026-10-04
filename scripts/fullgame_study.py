"""Experiment 15: the whole hand learned together -- multi-street lines, slowplaying, bluff timing, donk bets.

Scenarios (all at a 2.25bb open with the experiment 13 ranges; preflop/fullgame.py):
  base             BB checks every flop (as in experiments 8-14)
  donk             BB may also lead the flop for 1/3 pot (learned where and with what)
  donk_often       BB leads 50% of flops with every hand (forced); everything else adapts
  slowplay_X       BB calls a flop bet instead of raising with X% of its strong hands
                   (straights+, sets, two pair); everything else adapts. X = 0, 25, 50, 75
  no_early_bluffs  BB never bets or raises with weak hands (weak draws, overcards, air) on the
                   flop or turn, so its bluffs come preflop or on the river

Usage:  python scripts/fullgame_study.py            (~90 min on an idle multi-core machine; cached per scenario)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preflop import fullgame as fg
from preflop.cards import HANDS, n_combos
from preflop.flopgame import TEXTURES as FLOP_TEXTURES
from preflop.plots import grouped_bars, heatmap, line, stacked_bars

OPEN = 2.25
POT = 2 * OPEN
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"      # tiny end-to-end test run
BATCHES = 3 if QUICK else 250
HANDS_PER_BATCH = 15_000 if QUICK else 60_000
EVAL_HANDS = 15_000 if QUICK else 300_000
OUT = ROOT / "output" / ("fullgame_quick" if QUICK else "fullgame")
CACHE = OUT / "cache"                    # gitignored
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)
POLICIES = ROOT / "output" / "cbet_frequency" / "flop_policies.npz"
SLOWPLAY = [0.0, 0.25, 0.5, 0.75]
SCENARIOS = {"base": {"donk": False}, "donk": {"donk": True}, "donk_often": {"donk": True, "force_donk": 0.5},
             "no_early_bluffs": {"donk": True, "no_early_bluffs": True}}
SCENARIOS.update({f"slowplay_{int(x * 100)}": {"donk": True, "slowplay": x} for x in SLOWPLAY})
SCENARIOS["donk_rep"] = {"donk": True}               # training-noise check: re-learn "donk" with another seed
FEATURES = [("high card", 0, ["A-high", "K-high", "Q/J-high", "T-high or lower"]),
            ("pairing", 1, ["unpaired", "paired"]), ("suits", 2, ["rainbow", "two-tone", "monotone"]),
            ("connectedness", 3, ["disconnected", "connected"])]
GRAY, BLUE, ORANGE, AQUA = "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"


def run(name, opts, ranges):
    path = CACHE / f"{name}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return d
    print(f"== {name} {opts}", flush=True)
    t0 = time.time()
    seed = 2 if name.endswith("_rep") else 1          # "_rep" = same scenario, new training seed (noise check)
    sigma, _ = fg.learn(ranges["btn"], ranges["bb"], opts, s=OPEN, batches=BATCHES, hands_per_batch=HANDS_PER_BATCH,
                        seed=seed)
    value, acc = fg.evaluate(ranges["btn"], ranges["bb"], sigma, opts, s=OPEN, n=EVAL_HANDS)
    out = {"value_btn": value, "callchain": acc.callchain, "n_flop_bucket": acc.n_flop_bucket,
           "value_by_flop": acc.value_by_flop, "n_by_flop": acc.n_by_flop,
           "value_by_flop_bucket": acc.value_by_flop_bucket}
    for p in ("btn", "bb"):
        out[f"aggr_{p}"] = acc.aggr[p]
        out[f"decisions_{p}"] = acc.decisions[p]
    for (s, kind), v in acc.strat.items():
        out[f"strat_{s}_{kind}"] = v
        out[f"reach_{s}_{kind}"] = acc.reach[(s, kind)]
    np.savez_compressed(path, **out)
    print(f"   {name}: BB net result {OPEN - value:+.3f}bb per called pot "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return np.load(path)


def bb_value(d):
    """BB's net result for the whole hand (bb per called pot): pot share won minus everything it put in."""
    return OPEN - float(d["value_btn"])


def freq(d, s, kind, groups, axis_fn):
    """Reach-weighted action frequencies at a node, grouped by axis_fn(cell index arrays)."""
    strat, reach = d[f"strat_{s}_{kind}"], d[f"reach_{s}_{kind}"]
    out = {}
    for g, mask in groups(strat.shape).items():
        st = (strat * mask[..., None]).reshape(-1, strat.shape[-1]).sum(0)
        out[g] = st / max(float((reach * mask).sum()), 1e-12)
    return out


def by_bucket(shape, extra=None):
    groups = {}
    for b, name in enumerate(fg.BUCKETS):
        m = np.zeros(shape[:-1])
        m[:, :, b, ...] = 1
        groups[name] = m
    return groups


def by_flop_feature(shape, feature_idx, value, bucket=None):
    m = np.zeros(shape[:-1])
    for t, tex in enumerate(FLOP_TEXTURES):
        if tex[feature_idx] == value:
            if bucket is None:
                m[t] = 1
            else:
                m[t, :, bucket] = 1
    return m


def flop_value_by_feature(d):
    v, n = d["value_by_flop"], d["n_by_flop"]
    rows = {}
    for fname, k, values in FEATURES:
        for val in values:
            idx = [t for t, tex in enumerate(FLOP_TEXTURES) if tex[k] == val]
            rows[val] = OPEN - v[idx].sum() / max(n[idx].sum(), 1)      # BB net result on these flops
    return rows


def preflop_bluffs():
    """Share of the BB's hands it 3-bets as bluffs (hands with < 50% equity vs the BTN's opening range)."""
    from bb_study import solve_at
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    f = pd.read_csv(ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv", header=None,
                    index_col=0)[1]
    res, _ = solve_at(E, W, OPEN, float(f["alpha"]), float(f["r_oop"]))
    btn = res.btn_open[:, 1]
    eq_vs_open = ((1 - E) * W * btn[:, None]).sum(0) / (W * btn[:, None]).sum(0)   # BB hand j vs BTN range
    combos = np.array([n_combos(h) for h in HANDS])
    three = res.bb_vs_open[:, 2]
    bluff = eq_vs_open < 0.5
    total = combos.sum()
    return {"bb_3bet_pct": float((three * combos).sum() / total),
            "bb_3bet_bluff_pct": float((three * combos * bluff).sum() / total),
            "bluff_hands": [h for h, t, b in zip(HANDS, three, bluff) if t > 0.3 and b]}


def main():
    pol = np.load(POLICIES)
    ranges = {"btn": pol["btn_range"], "bb": pol["bb_range"]}
    res = {name: run(name, opts, ranges) for name, opts in SCENARIOS.items()}
    summary = pd.DataFrame([{"scenario": n, "bb_net_bb_per_called_pot": bb_value(d)} for n, d in res.items()])
    summary.round(4).to_csv(OUT / "scenario_values.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # 1. Multi-street lines: how often each BB flop hand check-calls the flop, then the turn, then the river.
    d = res["donk"]
    n = d["n_flop_bucket"]
    chain = pd.DataFrame({street: d["callchain"][s] / np.maximum(n, 1) for s, street in enumerate(fg.STREETS)},
                         index=fg.BUCKETS)
    chain["share_of_hands"] = n / n.sum()
    chain.round(3).to_csv(OUT / "bb_check_call_chain.csv")
    shown = chain[chain.share_of_hands > 0.005]
    grouped_bars(list(shown.index), {"check-calls the flop": list(shown.flop),
                                     "... and the turn": list(shown.turn), "... and the river": list(shown.river)},
                 "How often the BB check-calls street after street, by its flop hand", "Share of hands",
                 OUT / "bb_check_call_chain.png")

    # 2. Slowplay: BB value vs forced slowplay share, and the learned slowplay by hand.
    sp_vals = [bb_value(res[f"slowplay_{int(x * 100)}"]) for x in SLOWPLAY]
    free = freq(res["donk"], 0, "bb_vs_bet", by_bucket, None)
    learned_sp = {fg.BUCKETS[b]: float(free[fg.BUCKETS[b]][1] / max(free[fg.BUCKETS[b]][1] + free[fg.BUCKETS[b]][2], 1e-9))
                  for b in fg.STRONG}
    pd.DataFrame({"slowplay_share": SLOWPLAY, "bb_value": sp_vals}).round(4).to_csv(OUT / "slowplay_sweep.csv",
                                                                                     index=False)
    line([x * 100 for x in SLOWPLAY], [v * 1000 for v in sp_vals],
         "BB value vs how often it slowplays strong hands on the flop", "Slowplay share (% of strong hands that "
         "just call a flop bet)", "BB value (milli-bb per called pot)", OUT / "slowplay_sweep.png",
         highlight=(SLOWPLAY[int(np.argmax(sp_vals))] * 100, max(sp_vals) * 1000),
         highlight_label=f"best: slowplay {SLOWPLAY[int(np.argmax(sp_vals))]:.0%} ({max(sp_vals):+.3f}bb)")
    btn_free = freq(res["donk"], 0, "btn_vs_check", by_bucket, None)
    rows = []
    for b in fg.STRONG:
        name = fg.BUCKETS[b]
        f = free[name]
        rows.append({"hand": name, "bb_slowplays_vs_flop_bet": f[1] / max(f[1] + f[2], 1e-9),
                     "bb_fold": f[0], "bb_call": f[1], "bb_raise": f[2],
                     "btn_checks_back_flop": btn_free[name][0]})
    pd.DataFrame(rows).round(3).to_csv(OUT / "slowplay_by_hand.csv", index=False)

    # 3. Bluff timing: weak-hand aggression by street, with and without early bluffs; plus preflop 3-bet bluffs.
    pre = preflop_bluffs()
    brows = []
    for name in ("donk", "no_early_bluffs"):
        dd = res[name]
        hands = float(dd["n_flop_bucket"].sum())
        for p in ("bb", "btn"):
            aggr = dd[f"aggr_{p}"]
            for s, street in enumerate(fg.STREETS):
                weak = list(fg.WEAK) + ([8] if s == 2 else [])        # missed strong draws bluff on the river
                semi = [8] if s < 2 else []
                brows.append({"scenario": name, "player": p, "street": street,
                              "bluffs_per_hand": aggr[s, weak].sum() / hands,
                              "semi_bluffs_per_hand": aggr[s, semi].sum() / hands,
                              "value_bets_per_hand": aggr[s, :8].sum() / hands})
    bl = pd.DataFrame(brows)
    bl.round(4).to_csv(OUT / "bluffs_by_street.csv", index=False)
    pd.Series(pre | {"bluff_hands": " ".join(pre["bluff_hands"])}).to_csv(OUT / "preflop_bluffs.csv", header=False)
    for name in ("donk", "no_early_bluffs"):
        b = bl[(bl.scenario == name) & (bl.player == "bb")].set_index("street")
        stacked_bars(list(b.index), {"bluff": b.bluffs_per_hand, "semi-bluff (draw)": b.semi_bluffs_per_hand,
                                     "value": b.value_bets_per_hand},
                     f"BB bets / raises per called pot, by street ({name.replace('_', ' ')})", "Bets per hand",
                     OUT / f"bb_aggression_by_street_{name}.png",
                     colors={"bluff": ORANGE, "semi-bluff (draw)": AQUA, "value": BLUE})

    # 4. Donk bets: where the BB leads (learned), and what leading pays by board, learned vs forced.
    donk_rows = []
    lead = res["donk"]["strat_0_bb_first"]
    reach = res["donk"]["reach_0_bb_first"]
    base_v = flop_value_by_feature(res["base"])
    donk_v = flop_value_by_feature(res["donk"])
    often_v = flop_value_by_feature(res["donk_often"])
    for fname, k, values in FEATURES:
        for val in values:
            m = by_flop_feature(reach.shape + (1,), k, val)
            r = (reach * m).sum()
            donk_rows.append({"feature": fname, "value": val, "bb_donks_learned": (lead[..., 1] * m).sum() / r,
                              "bb_value_no_donk": base_v[val], "bb_value_learned_donk": donk_v[val],
                              "bb_value_donk_50pct": often_v[val],
                              "gain_from_donk_option": donk_v[val] - base_v[val],
                              "change_from_donking_50pct": often_v[val] - base_v[val]})
    dk = pd.DataFrame(donk_rows)
    dk.round(4).to_csv(OUT / "donk_by_texture.csv", index=False)
    grid = {}
    for fname, k, values in FEATURES:
        for val in values:
            row = []
            for b in range(fg.NB):
                m = by_flop_feature(reach.shape + (1,), k, val, bucket=b)
                r = (reach * m).sum()
                row.append((lead[..., 1] * m).sum() / r if r > (300 if not QUICK else 5) else np.nan)
            grid[val] = row
    g = pd.DataFrame(grid, index=fg.BUCKETS).T
    g = g.loc[:, g.notna().any()]
    g.round(3).to_csv(OUT / "donk_heatmap.csv")
    heatmap(g, "How often the BB donks (leads 1/3 pot) the flop: board feature x hand", "BB hand on the flop",
            "Board feature", OUT / "donk_heatmap.png", fmt="pct", figsize=(10, 6))
    grouped_bars(list(dk.value), {"BB donks (learned)": list(dk.bb_donks_learned)},
                 "How often the BB donks the flop, by board", "Donk frequency", OUT / "donk_by_board.png",
                 groups=[(f, len(v)) for f, _, v in FEATURES])
    grouped_bars(list(dk.value), {"learned donking vs never": list(dk.gain_from_donk_option * 10),
                                  "donk 50% of flops vs never": list(dk.change_from_donking_50pct * 10)},
                 "Change in BB value from donking, by board (bb per 10 called pots)", "bb per 10 called pots",
                 OUT / "donk_value_by_board.png", groups=[(f, len(v)) for f, _, v in FEATURES], pct=False)
    print(chain.round(2).to_string())
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(bl.round(4).to_string(index=False))
    print(pre)
    print(dk.round(3).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
