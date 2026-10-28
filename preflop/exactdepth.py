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
"""
import json

import numpy as np

from preflop.exactcoverage import CoverageTrainer
from preflop.exactgame import SEATS
from preflop.exacttrain import _metadata

LEAF_STREET = 1                                    # the flop: the first postflop decision is the legacy engine's
ENGINE = {"exact_required": False, "pot_odds": True, "exact_policy_path": False}


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

    def __init__(self, path, config=None, targets=None, **kwargs):
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
        except BaseException:
            self.close()
            raise

    def _walk(self, state, updater, own=1.0, opponent=1.0, sampled=1.0):
        if not state.terminal and state.street >= LEAF_STREET:
            return self._leaf(state, updater)
        return super()._walk(state, updater, own, opponent, sampled)

    def _leaf(self, state, updater):
        """One rollout of the legacy engine for both seats to the end of the deal's runout (updater's result)."""
        seed = int(self.rng.integers(2 ** 63 - 1))
        for engine in self._engines:
            engine.rng = np.random.default_rng(seed)
        while not state.terminal:
            d = self._engines[state.actor].decide(SEATS[state.actor], "".join(state.holes[state.actor]),
                                                  "".join(state.board), _history(state),
                                                  stack=state.stack * state.config.chip_bb)
            state = state.step(_label(state, d.action, d.to), observed=True)
        self.aggregates["leaf_rollouts"] = self.aggregates.get("leaf_rollouts", 0) + 1
        return state.utility(updater)
