"""Depth-limited exact preflop training: exact preflop nodes, the frozen legacy engine from the flop on.

Why (2026-10-27): full-game coverage training left 301 of 357 preflop targets unstable (average-strategy TV 0.24 per
update doubling at ~16k updates; current strategies like A4o BTN jamming 25%). One sampled hand below a preflop node
runs through millions of postflop infosets that are almost never revisited, so the value it returns is mostly noise.
Neither linear averaging nor VR-MCCFR baselines changed that (A/B in output/exact/vr-ab-2026-10-27/).

Here the walk stops when the flop is dealt and values the leaf with one rollout of the frozen legacy engine for both
seats over the deal's own runout. Preflop all-ins still end at the exact showdown. The leaf value is an unbiased
sample of "both players continue with the legacy postflop policy", so the preflop strategy is trained for that
continuation, not for an opponent who deviates after the flop (not an exploitability claim).

The continuation (engine settings, bot/preflop source and policy-file fingerprints) is recorded when the candidate is
created; resuming with a different continuation raises.

Sampling (2026-10-28): the outcome-sampling walk stayed flat (preflop TV 0.21 from 512 to 2048 updates per target, 9/357
ready after ~3h). Option "external" walks every action at the updating player's preflop nodes (the opponent and chance
stay sampled, so no 1/q weight on the updater's own action) and values each leaf by the mean of leaf_runouts rollouts,
each over a freshly drawn runout (chance below the leaf, so still unbiased). Recorded as meta "depth_sampling".
"""
import json
from dataclasses import replace

import numpy as np

from preflop.exactcoverage import CoverageTrainer
from preflop.exactgame import DECK, SEATS
from preflop.exacttrain import _metadata

LEAF_STREET = 1                                    # the flop: the first postflop decision is the legacy engine's
ENGINE = {"exact_required": False, "pot_odds": True, "exact_policy_path": False}
SAMPLING = ("outcome", "external")


def continuation_fingerprint(engine):
    from bot.audit import ROOT, _fingerprint, policy_files
    sources = [p for folder in ("bot", "preflop") for p in (ROOT / folder).rglob("*.py")]
    return {"kind": "legacy_rollout", "leaf_street": LEAF_STREET, "engine": ENGINE,
            "source_sha256": _fingerprint(sources, ROOT), "policy_sha256": _fingerprint(policy_files(engine), ROOT)}


def _history(state):
    return [[[SEATS[p], kind, to * state.config.chip_bb] for p, kind, to in street] for street in state.history]


def _label(state, action, to):
    """A legacy engine action as a legal game label (sizes clipped to the legal range, as in live play)."""
    if action in ("bet", "raise"):
        if not any(a.startswith("to:") for a in state.actions()):
            return "call" if state.facing else "check"
        return f"to:{min(state.cap, max(state.minimum, round(to / state.config.chip_bb)))}"
    if action in ("call", "check") or not state.facing:
        return "call" if state.facing else "check"          # nothing to call: a fold is a check
    return "fold"


class DepthLimitedTrainer(CoverageTrainer):
    depth_limited = True

    def __init__(self, path, config=None, targets=None, sampling=None, leaf_runouts=None, **kwargs):
        if targets is not None and any((t.get("context", t)).get("board") for t in targets):
            raise ValueError("depth-limited training takes preflop targets only")
        from bot.engine import Engine
        self._engines = tuple(Engine(seed=0, **ENGINE) for _ in SEATS)
        record = continuation_fingerprint(self._engines[0])
        super().__init__(path, config, targets=targets, **kwargs)
        try:
            meta = _metadata(self.db)
            if "continuation" not in meta:
                if meta["iterations"]:
                    raise ValueError("cannot add a depth limit to a candidate that already trained the full game")
                self._set("continuation", record)
                self.db.commit()
            elif meta["continuation"] != record:
                raise ValueError("continuation changed (legacy engine source, policies or settings); "
                                 "start a new candidate")
            if any(json.loads(text)["board"] for (text,) in self.db.execute("SELECT context FROM coverage_targets")):
                raise ValueError("depth-limited candidate holds postflop targets")
            requested = None if sampling is None and leaf_runouts is None else \
                {"kind": sampling or "outcome", "leaf_runouts": leaf_runouts or 1}
            recorded = meta.get("depth_sampling")
            if recorded is None:
                if meta["iterations"] and requested not in (None, {"kind": "outcome", "leaf_runouts": 1}):
                    raise ValueError("cannot change the sampling of a candidate that already trained")
                recorded = requested or {"kind": "outcome", "leaf_runouts": 1}
                if recorded["kind"] not in SAMPLING or int(recorded["leaf_runouts"]) < 1:
                    raise ValueError("unknown depth-limited sampling")
                self._set("depth_sampling", recorded)
                self.db.commit()
            elif requested is not None and requested != recorded:
                raise ValueError("depth-limited sampling differs from the one recorded with the candidate")
            self.sampling = recorded
        except BaseException:
            self.close()
            raise

    def _walk(self, state, updater, own=1.0, opponent=1.0, sampled=1.0):
        if state.terminal:
            return state.utility(updater)
        if state.street >= LEAF_STREET:
            return self._leaf(state, updater)
        if self.sampling["kind"] == "outcome" or state.actor != updater:
            return super()._walk(state, updater, own, opponent, sampled)
        # External sampling at the updater's own node: every action's value, no importance weight for this choice.
        key, actions, row, strategy = self._node(state)
        values = np.array([self._walk(state.step(a), updater, own * strategy[i], opponent, sampled)
                           for i, a in enumerate(actions)])
        value = float(strategy @ values)
        delta, average = (opponent / sampled) * (values - value), (own / sampled) * strategy
        if not np.isfinite(delta).all() or not np.isfinite(average).all():
            raise FloatingPointError("non-finite importance weighted update; batch rolled back")
        self._record(state, key, actions, row, delta, average, True)
        return value

    def _leaf(self, state, updater):
        """Mean over leaf_runouts rollouts of the legacy engine for both seats (updater's result). With one rollout the
        deal's own runout is used; with more, each rollout draws a fresh five-card runout (no preflop decision saw the
        flop, so this is still chance below the leaf)."""
        n = self.sampling["leaf_runouts"]
        total = 0.0
        for _ in range(n):
            leaf = state
            if n > 1:
                rest = [c for c in DECK if c not in set(state.holes[0]) | set(state.holes[1])]
                leaf = replace(state, runout=tuple(rest[i] for i in self.rng.choice(len(rest), size=5, replace=False)))
            seed = int(self.rng.integers(2 ** 63 - 1))
            for engine in self._engines:
                engine.rng = np.random.default_rng(seed)
            while not leaf.terminal:
                d = self._engines[leaf.actor].decide(SEATS[leaf.actor], "".join(leaf.holes[leaf.actor]),
                                                     "".join(leaf.board), _history(leaf),
                                                     stack=leaf.stack * leaf.config.chip_bb)
                leaf = leaf.step(_label(leaf, d.action, d.to), observed=True)
            total += leaf.utility(updater)
        self.aggregates["leaf_rollouts"] = self.aggregates.get("leaf_rollouts", 0) + n
        return total / n
