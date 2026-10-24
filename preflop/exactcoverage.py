"""Guided full-game MCCFR for queued literal-card/depth/action-history cells.

Chance and action proposals retain full support and use importance corrections.
Every trajectory starts at the blinds; no artificial subgame range is installed.
"""
import hashlib
import json
from dataclasses import replace
from math import comb

import numpy as np

from preflop.exactgame import SEATS, State, cards, from_context
from preflop.exacttrain import Trainer, _metadata

TARGET_SHARE = .8
HISTORY_SHARE = .9


def chance_weight(matches: bool, compatible_combos: int) -> float:
    p = 1 / compatible_combos
    q = (TARGET_SHARE if matches else 0) + (1 - TARGET_SHARE) * p
    return p / q


class CoverageTrainer(Trainer):
    def __init__(self, path, config=None, targets=None, seed=0):
        contexts = [r.get("context", r) for r in targets] if targets is not None else None
        super().__init__(path, config, seed=seed)
        try:
            self.db.execute("CREATE TABLE IF NOT EXISTS coverage_targets (key TEXT PRIMARY KEY, context TEXT NOT NULL)")
            meta = _metadata(self.db)
            if "coverage_manifest" not in meta:
                if not contexts or meta["iterations"]:
                    raise ValueError("coverage training needs a new candidate and nonempty exact targets")
                canonical = {}
                for context in contexts:
                    state = from_context(self.config, context, np.random.default_rng(0), known_hole=True)
                    clean = {"hero_position": SEATS[state.actor], "hero_hole": "".join(state.holes[state.actor]),
                             "board": "".join(state.board), "effective_stack_bb": state.stack * self.config.chip_bb,
                             "history": [[[SEATS[p], k, to * self.config.chip_bb] for p, k, to in st] for st in state.history]}
                    canonical[state.key()] = json.dumps(clean, sort_keys=True)
                self.db.executemany("INSERT INTO coverage_targets VALUES (?,?)", sorted(canonical.items()))
                manifest = hashlib.sha256(json.dumps(sorted(canonical.items())).encode()).hexdigest()
                self._set("coverage_manifest", manifest)
                self._set("coverage_prior", "whole_game_from_blinds_importance_corrected")
                self._set("coverage_sampling", {"target_share": TARGET_SHARE, "history_share": HISTORY_SHARE, "correction": "p_over_q"})
                self.db.commit()
            stored = self.db.execute("SELECT key,context FROM coverage_targets ORDER BY key").fetchall()
            digest = hashlib.sha256(json.dumps(stored).encode()).hexdigest()
            if digest != _metadata(self.db)["coverage_manifest"]:
                raise ValueError("coverage target manifest integrity mismatch")
            self.targets = [(key, json.loads(text)) for key, text in stored]
            if not self.targets:
                raise ValueError("empty coverage target manifest")
            if contexts is not None and "coverage_manifest" in meta:
                requested = set()
                for c in contexts:
                    state = from_context(self.config, c, np.random.default_rng(0), known_hole=True)
                    requested.add(state.key())
                if requested != {key for key, _ in self.targets}:
                    raise ValueError("coverage target manifest changed")
            self.target_actions = {key: from_context(self.config, c, np.random.default_rng(0), known_hole=True).actions()
                                   for key, c in self.targets}
            self._draw = meta["iterations"] * 2
        except BaseException:
            self.close()
            raise

    def _sample(self, updater):
        _, context = self.targets[(self._draw // 2) % len(self.targets)]
        self._draw += 1
        self._guide = from_context(self.config, context, np.random.default_rng(0), known_hole=True)
        targeted = self.rng.random() < TARGET_SHARE
        root = State.deal(self.config, self.rng, stack_bb=context["effective_stack_bb"],
                          hole=context["hero_hole"] if targeted else None,
                          board=context["board"] if targeted else ())
        hero = SEATS.index(context["hero_position"])
        if targeted and hero == 1:
            root = replace(root, holes=(root.holes[1], root.holes[0]))
        count = len(cards(context["board"]))
        board = tuple(sorted(root.runout[:3])) + root.runout[3:count] if count else ()
        wanted = cards(context["board"])
        wanted = tuple(sorted(wanted[:3])) + wanted[3:]
        matches = set(root.holes[hero]) == set(cards(context["hero_hole"])) and board == wanted
        # Marginal event probability: literal hero combo AND visible board.
        outcomes = comb(52, 2)
        if count >= 3:
            outcomes *= comb(50, 3)
        if count >= 4:
            outcomes *= 47
        if count >= 5:
            outcomes *= 46
        return root, chance_weight(matches, outcomes)

    def _behavior(self, state, updater, strategy, actions):
        behavior = super()._behavior(state, updater, strategy, actions)
        guide = self._guide
        if state.street > guide.street:
            return behavior
        # Guide only a matching observed prefix; never truncate repeated raises.
        if any(actual != target[:len(actual)] for actual, target in zip(state.history, guide.history)):
            return behavior
        target = guide.history[state.street]
        if len(state.history[state.street]) >= len(target):
            return behavior
        _, kind, to = target[len(state.history[state.street])]
        action = f"to:{to}" if kind in ("bet", "raise") else kind
        if action not in actions:
            raise ValueError("observed raise missing from configured training menu")
        behavior = (1 - HISTORY_SHARE) * behavior
        behavior[actions.index(action)] += HISTORY_SHARE
        return behavior

    def train(self, iterations):
        self._draw = _metadata(self.db)["iterations"] * 2
        return super().train(iterations)

    def coverage(self, minimum=1000):
        if minimum < 1:
            raise ValueError("minimum updates must be positive")
        by_street, by_stack, by_raises = {}, {}, {}
        trained, floor, lowest, updates = 0, 0, None, []
        for key, c, n, actions, total in self.db.execute("SELECT c.key,c.context,coalesce(n.updates,0),n.actions,n.total FROM coverage_targets c "
                                       "LEFT JOIN nodes n ON n.key=c.key"):
            info, context = json.loads(key), json.loads(c)
            floor += int(n >= minimum)
            ready = False
            if n >= minimum:
                try:
                    labels, sums = tuple(json.loads(actions)), np.asarray(json.loads(total), float)
                    ready = (labels == self.target_actions[key] and sums.shape == (len(labels),)
                             and np.isfinite(sums).all() and (sums >= 0).all()
                             and np.isfinite(sums.sum()) and sums.sum() > 0)
                except (TypeError, ValueError):
                    pass
            trained += int(ready)
            updates.append(n)
            lowest = n if lowest is None else min(lowest, n)
            for group, name in ((by_street, str(info["street"])),
                                (by_stack, str(context["effective_stack_bb"])),
                                (by_raises, str(sum(k in ("bet", "raise") for st in context["history"] for _, k, _ in st)))):
                entry = group.setdefault(name, {"targets": 0, "trained": 0})
                entry["targets"] += 1
                entry["trained"] += int(ready)
        return {"targets": len(self.targets), "trained_targets": trained, "update_floor_targets": floor, "minimum_updates": minimum,
                "minimum_observed_updates": lowest or 0, "maximum_observed_updates": max(updates, default=0),
                "fraction": trained / len(self.targets), "by_street": by_street, "by_stack_bb": by_stack,
                "by_prior_raises": by_raises, "full_game_complete": False,
                "scope": "queued_exact_infosets_only", "live_eligible": False,
                "reason": "requires_frozen_evaluation_and_promotion"}
