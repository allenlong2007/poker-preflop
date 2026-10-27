"""Where does the exact candidate lose at its live contexts? (diagnosis for scripts/live_context_eval.py)

The live-context pilot (-0.90 bb/visit uniform, -2.13 posterior) compares the candidate hero with the legacy hero.
Almost every exact decision in it is the root decision (145,886 exact decisions over 145,808 served roots), so the
loss is made at the root. This script prices every root action the candidate or the legacy policy would take:

    Q(a) = hero's result after taking `a` at the root, then the legacy hero vs the villain to the end (same seed for
           every a, so all actions see the same runout and villain cards)
    gap  = sum_a (p_candidate(a) - p_legacy(a)) * Q(a)        the root-only loss, with no sampling noise at the root

and splits the gap by
    villain   legacy (posterior-resampled deals, as in the pilot) and held-out bots (uniform deals): does the exact
              policy only lose to the legacy villain the legacy hero was tuned against?
    action    which root actions the candidate over- or under-uses, and what that costs
    updates   training updates of the served node: if the loss shrinks with updates, it is undertraining noise
    continuation  whether the hero acts again after the root action: a loss on actions with no later hero decision
              (folds, river calls, all-ins) cannot come from the exact line being abandoned by the legacy continuation

Usage: python scripts/live_context_diagnose.py --path ready.sqlite --targets targets.json --output diag.json
       [--deals 40] [--draws 1000] [--seed 4000037] [--villains legacy,geo,trapper,limper,threebet,base]
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_context_eval import _history, _reseed, _weight, load_contexts  # noqa: E402
from preflop.exactgame import Config, SEATS, from_context  # noqa: E402
from preflop.exacttrain import _runtime, file_hash  # noqa: E402

EPS = 1e-9
_W = {}


def _init(path, config, villains):
    from bot.engine import Engine
    from bot.heldout import make
    _W["candidate"] = Engine(exact_required=False, seed=0, pot_odds=True, exact_policy_path=path,
                             exact_candidate=True, exact_min_updates=1000)
    _W["hero"] = Engine(exact_required=False, seed=0, pot_odds=True, exact_policy_path=False)
    _W["legacy"] = Engine(exact_required=False, seed=0, pot_odds=True, exact_policy_path=False)
    _W["bots"] = {v: make(v) for v in villains if v != "legacy"}
    _W["config"] = Config(**config)


def _label(state, kind, to):
    """An engine or bot action as a game label, with the same clipping as the evaluation."""
    if kind in ("bet", "raise", "all-in"):
        if not any(a.startswith("to:") for a in state.actions()):
            return "call" if state.facing else "check"
        return f"to:{min(state.cap, max(state.minimum, round(to / state.config.chip_bb)))}"
    if kind in ("call", "check"):
        return "call" if state.facing else "check"
    return "fold"


def _policy(engine, state):
    """The engine's root probabilities over game labels (sample=False: no sampling at the root)."""
    d = engine.decide(SEATS[state.actor], "".join(state.holes[state.actor]), "".join(state.board),
                      _history(state), sample=False, stack=state.stack * state.config.chip_bb)
    out = {}
    for label, p in d.probs.items():
        if p > EPS:
            kind, to = engine._options[label]
            g = _label(state, kind, to)
            out[g] = out.get(g, 0.0) + p
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}, d


class _BotSeat:
    """A held-out bot playing a seat of the exact game."""

    def __init__(self, bot):
        self.bot = bot

    def decide(self, state):
        from bot.engine import Act
        hist = [[Act(SEATS[p], k, to * state.config.chip_bb) for p, k, to in street] for street in state.history]
        self.bot.stack = state.stack * state.config.chip_bb
        import eval7
        hole, board = [eval7.Card(str(c)) for c in state.holes[state.actor]], [eval7.Card(str(c)) for c in state.board]
        kind, to = self.bot.act(SEATS[state.actor], hole, board, hist,
                                state.facing * state.config.chip_bb)
        return _label(state, kind, to)


def _finish(state, hero_seat, villain, seed):
    """Legacy hero vs `villain` from `state` to the end: hero's result and whether the hero acted again."""
    hero = _W["hero"]
    _reseed([hero, _W["legacy"]], seed)
    if isinstance(villain, _BotSeat):
        villain.bot.rng = np.random.default_rng(seed)
    acted = False
    while not state.terminal:
        if state.actor == hero_seat or not isinstance(villain, _BotSeat):
            engine = hero if state.actor == hero_seat else villain
            d = engine.decide(SEATS[state.actor], "".join(state.holes[state.actor]), "".join(state.board),
                              _history(state), stack=state.stack * state.config.chip_bb)
            label = _label(state, d.action, d.to)
            acted |= state.actor == hero_seat
        else:
            label = villain.decide(state)
        state = state.step(label, observed=True)
    return state.utility(hero_seat), acted


def run_target(job):
    index, context, deals, draws, seed, villains = job
    config, candidate, legacy = _W["config"], _W["candidate"], _W["legacy"]
    hero_seat = SEATS.index(context["hero_position"])
    rows = []
    for villain_name in villains:
        if villain_name == "legacy":
            # The pilot's posterior: uniform draws resampled by the legacy villain's probability of its line.
            seeds = [int(np.random.SeedSequence([seed, index, k]).generate_state(1)[0]) for k in range(draws)]
            roots = [from_context(config, context, np.random.default_rng(s), known_hole=True) for s in seeds]
            w = np.array([_weight(r, legacy, context, (candidate, _W["hero"], legacy), s) for r, s in zip(roots, seeds)])
            if w.sum() <= 0:
                continue
            u = (np.random.default_rng([seed, index, 1]).random() + np.arange(deals)) / deals
            picked = np.searchsorted(np.cumsum(w / w.sum()), u, side="right").clip(0, draws - 1)
            sample = [(seeds[k], roots[k]) for k in picked]
            villain = legacy
        else:
            seeds = [int(np.random.SeedSequence([seed, index, 7, k]).generate_state(1)[0]) for k in range(deals)]
            sample = [(s, from_context(config, context, np.random.default_rng(s), known_hole=True)) for s in seeds]
            villain = _BotSeat(_W["bots"][villain_name])
        for s, root in sample:
            _reseed([candidate, _W["hero"], legacy], s)
            pc, d = _policy(candidate, root)
            if d.source != "exact":
                continue                                       # not served: the candidate plays legacy here
            updates = d.diagnostics["exact"].get("updates")
            pl, _ = _policy(_W["hero"], root)
            q, acted = {}, {}
            for label in set(pc) | set(pl):
                q[label], acted[label] = _finish(root.step(label, observed=True), hero_seat, villain, s)
            rows.append({"villain": villain_name, "updates": updates, "pc": pc, "pl": pl, "q": q, "acted": acted,
                         "pot": root.pot * config.chip_bb, "facing": root.facing * config.chip_bb,
                         "cap": root.cap, "street": len(root.board)})
    return index, rows


def category(label, row):
    if label in ("fold", "check", "call"):
        return label
    if int(label[3:]) == row["cap"]:
        return "all-in"
    return "bet/raise"


def summarise(results, contexts):
    """Per-target means of the root gap, then target-level mean and SE, split several ways."""
    def stat(values):
        v = np.asarray(values, float)
        if len(v) < 2:
            return {"targets": len(v)}
        se = float(v.std(ddof=1) / math.sqrt(len(v)))
        return {"targets": len(v), "mean_bb": round(float(v.mean()), 3), "se_bb": round(se, 3),
                "z": round(float(v.mean() / se), 1) if se > 0 else None}

    by = {}

    def add(key, index, value):
        by.setdefault(key, {}).setdefault(index, []).append(value)

    for index, rows in results.items():
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(contexts[index]["board"]) // 2]
        for r in rows:
            v = r["villain"]
            gap = sum((r["pc"].get(a, 0) - r["pl"].get(a, 0)) * r["q"][a] for a in r["q"])
            add(("gap", v), index, gap)
            add(("gap_street", v, street), index, gap)
            u = r["updates"] or 0
            bucket = "1k-2k" if u < 2000 else "2k-10k" if u < 10000 else "10k+"
            add(("gap_updates", v, bucket), index, gap)
            tv = 0.5 * sum(abs(r["pc"].get(a, 0) - r["pl"].get(a, 0)) for a in r["q"])
            add(("tv", v), index, tv)
            cats, cont = {}, {True: 0.0, False: 0.0}
            for a in r["q"]:
                c = (r["pc"].get(a, 0) - r["pl"].get(a, 0)) * r["q"][a]
                k = category(a, r)
                cats[k] = cats.get(k, 0.0) + c
                cont[r["acted"][a]] += c
            for k in ("fold", "check", "call", "bet/raise", "all-in"):
                add(("gap_action", v, k), index, cats.get(k, 0.0))
                add(("use_candidate", v, k), index, sum(p for a, p in r["pc"].items() if category(a, r) == k))
                add(("use_legacy", v, k), index, sum(p for a, p in r["pl"].items() if category(a, r) == k))
            add(("gap_hero_acts_again", v, True), index, cont[True])
            add(("gap_hero_acts_again", v, False), index, cont[False])
    out = {}
    for key, per_target in by.items():
        means = [float(np.mean(x)) for x in per_target.values()]
        if key[0] in ("tv", "use_candidate", "use_legacy"):
            out["|".join(map(str, key))] = round(float(np.mean(means)), 3)
        else:
            out["|".join(map(str, key))] = stat(means)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deals", type=int, default=40)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=4_000_037)
    p.add_argument("--villains", default="legacy,geo,trapper,limper,threebet,base")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    villains = args.villains.split(",")
    meta, contexts, digest = load_contexts(args.path, args.targets, args.seed)
    contexts = contexts[:args.limit]
    fingerprint = file_hash(args.path)
    jobs = [(i, c, args.deals, args.draws, args.seed, villains) for i, c in enumerate(contexts)]
    results = {}
    with Pool(args.workers, initializer=_init, initargs=(str(args.path), meta["config"], villains)) as pool:
        for n, (index, rows) in enumerate(pool.imap_unordered(run_target, jobs), 1):
            results[index] = rows
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} targets", flush=True)
    if file_hash(args.path) != fingerprint:
        raise ValueError("candidate changed during diagnosis")
    report = {"diagnosis": "live_context_root_gap", "candidate_sha256": fingerprint, "coverage_manifest": digest,
              "runtime": _runtime(), "seed": args.seed, "deals": args.deals, "draws": args.draws,
              "villains": villains, "summary": summarise(results, contexts)}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    rows_path = args.output.with_suffix(".rows.json")
    rows_path.write_text(json.dumps({str(k): v for k, v in results.items()}))
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
