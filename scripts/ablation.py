"""Experiment 50: does each adjustment on top of the base policy help? Ablation matches against held-out bots.

Trapping, river jams, calling corrections and opponent adjustments modify the base policy one after another, so
the solver's accuracy on the underlying trees says nothing about the final bot. Each stage is measured by
duplicate matches on fixed deals against held-out opponents (bot/heldout.py, never used for tuning):

    full        every stage on (the default engine)
    base        every stage off
    -stage      full minus one stage (what the stage adds given the others: leave-one-out)
    +stage      base plus one stage (what it adds alone)
    read200     full plus a 200-deal read of the opponent (bot/adapt.py adaptation, 100bb only)
    read1000    the same after 1000 deals
    potodds_all   the pot-odds check at every depth (before experiment 50 limited it to POT_ODDS_MAX_BB)
    potodds_none  no pot-odds check at any depth

Every variant plays the same deals with the same engine and opponent seeds, so results are compared per deal
(paired): the reported difference to `full` has a much smaller error than either result alone. Also reports how
often each stage changed a decision in `full`.

Usage: python scripts/ablation.py [deals] [stacks e.g. 100,20] [opponents] [variants]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "output" / "match" / "ablation"
DEAL_SEED = int(__import__("os").environ.get("DEAL_SEED", 50_001))   # deals no tuning run used (3, 7 + stack)
READS = {"read200": 200, "read1000": 1000}
STAGES = ("vs_shove", "small_raise", "reraise_war", "monster", "trap", "river_jam", "short_stack", "pot_odds",
          "oop_correction", "small_leads", "polar_river")
VARIANTS = ("full", "base") + tuple(f"-{s}" for s in STAGES) + tuple(f"+{s}" for s in STAGES) + tuple(READS)


def engine(variant):
    from bot.engine import Engine
    e = Engine(exact_required=False, seed=1, pot_odds=True)
    if variant == "potodds_all":
        e.pot_odds_max_bb = float("inf")
    elif variant == "potodds_none":
        e.off = {"pot_odds"}
    elif variant == "base":
        e.off = set(STAGES)
    elif variant.startswith("-"):
        e.off = {variant[1:]}
    elif variant.startswith("+"):
        e.off = set(STAGES) - {variant[1:]}
    return e


class Agent:
    """The bot, counting which stages changed its decisions."""

    def __init__(self, eng, stack):
        self.engine, self.stack, self.fired, self.decisions = eng, stack, {}, 0

    def act(self, seat, hole, board, history, facing):
        d = self.engine.decide(seat, hole, board, history, stack=self.stack)
        self.decisions += 1
        for a in d.diagnostics.get("adjustments", ()):
            self.fired[a["stage"]] = self.fired.get(a["stage"], 0) + 1
        return d.action, d.to


def read(job):
    """Opponent state from `deals` deals of its play (other deals than the match), through the live adapter's
    evidence gates (bot.adapt.read_state)."""
    import strictness_train as stt
    from bot.adapt import BASELINE, Preflop, read_state
    from bot.heldout import make
    opp, deals = job
    stt.make = make
    recs = stt.profile_of(opp, deals)["records"]
    for i, r in enumerate(recs):              # identities the gates require (they fail closed without them)
        r.update(id=f"read-{i}", opp_id=f"heldout-{opp}", hero_id="bot")
    state, pre, _ = read_state(recs, json.load(open(BASELINE)), Preflop())
    return job, state


def apply_read(e, st):
    """As bot.adapt.Adapter loads an opponent's state into the engine before their decisions."""
    e.set_adaptation(st["nodes"], st["lam"], st["post"])
    e.opp_profile, e.size_value, e.strict = st["profile"], st["size_value"], st["strict"]
    e.bluff_model = st["bluff_model"]
    e.read_evidence = st["evidence"]


def run(job):
    variant, stack, opp, deals = job
    from bot.heldout import make
    from bot.selfplay import DECK, play_hand
    e = engine(variant)
    if variant in READS:
        apply_read(e, read((opp, READS[variant]))[1])
    hero, villain = Agent(e, stack), make(opp)
    villain.stack = stack
    rng = np.random.default_rng(DEAL_SEED + int(stack))
    out = np.zeros(deals)
    for k in range(deals):
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        holes, board = {"btn": c[:2], "bb": c[2:4]}, c[4:]
        for hero_seat, sign in (("btn", 1), ("bb", -1)):
            agents = {"btn": hero, "bb": villain} if hero_seat == "btn" else {"btn": villain, "bb": hero}
            log = []
            out[k] += sign * play_hand(agents, holes, board, stack=stack, log=log)
            if hasattr(villain, "observe"):
                villain.observe("bb" if hero_seat == "btn" else "btn", log, holes, board)
    return (variant, stack, opp), out / 2 * 100, hero.fired, hero.decisions     # bb/100 per deal (both seats)


def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    stacks = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [100.0, 20.0]
    from bot.heldout import HELDOUT
    opps = sys.argv[3].split(",") if len(sys.argv) > 3 else list(HELDOUT)
    variants = sys.argv[4].split(",") if len(sys.argv) > 4 else list(VARIANTS)
    with Pool(10) as pool:
        jobs = [(v, s, o, deals) for v in variants for s in stacks for o in opps if not (v in READS and s != 100)]
        res = {key: (r, fired, n) for key, r, fired, n in pool.imap_unordered(run, jobs)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "per_deal.npz", **{"|".join(map(str, k)): r for k, (r, _, _) in res.items()})
    summary = {}
    print(f"bb/100 vs held-out bots ({deals} duplicate deals per cell; differences are paired per deal vs full)")
    for s in stacks:
        print(f"\n-- {s:g}bb effective")
        print("   variant          " + " ".join(f"{o:>12}" for o in opps) + "        mean diff (z)")
        for v in variants:
            if (v, s, opps[0]) not in res:
                continue
            cells, diffs = [], []
            for o in opps:
                r = res[(v, s, o)][0]
                if v == "full":
                    cells.append(f"{r.mean():+7.1f} ±{r.std(ddof=1) / np.sqrt(len(r)):3.0f}")
                    continue
                dlt = r - res[("full", s, o)][0]
                diffs.append(dlt)
                cells.append(f"{dlt.mean():+7.1f} ±{dlt.std(ddof=1) / np.sqrt(len(dlt)):3.0f}")
            line = f"   {v:16} " + " ".join(f"{c:>12}" for c in cells)
            if diffs:
                pooled = np.concatenate(diffs)
                m, se = pooled.mean(), pooled.std(ddof=1) / np.sqrt(len(pooled))
                line += f"   {m:+6.1f} ({m / se:+.1f})" if se > 0 else "     0.0 (never changed a result)"
                summary[f"{v}|{s:g}"] = {"diff": float(m), "se": float(se),
                                         "by_opponent": {o: [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))]
                                                         for o, d in zip(opps, diffs)}}
            else:
                summary[f"full|{s:g}"] = {o: float(res[("full", s, o)][0].mean()) for o in opps}
            print(line)
        fired = {}
        n = 0
        for o in opps:
            _, f, k = res[("full", s, o)]
            n += k
            for st, c in f.items():
                fired[st] = fired.get(st, 0) + c
        print("   stages changing a decision in full (share of decisions): " +
              ", ".join(f"{st} {c / n:.1%}" for st, c in sorted(fired.items(), key=lambda x: -x[1])))
        summary[f"fired|{s:g}"] = {st: c / n for st, c in fired.items()}
    print("\n   -stage: negative = removing the stage loses (the stage helps). +stage: positive = the stage helps on "
          "its own.\n   |z| > 3.3 is significant after a Bonferroni correction over every stage test at both depths.")
    (OUT / "summary.json").write_text(json.dumps({"deals": deals, "deal_seed": DEAL_SEED, "summary": summary},
                                                 indent=1))


if __name__ == "__main__":
    main()
