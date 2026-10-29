"""Whole preflop game on a stack grid: hand classes, a fitted legacy leaf model, and serving by nearest grid stack.

Why (2026-10-29): literal live targets (one hand, one exact stack and history each) almost never recur live (317 of
357 logged spots were unique) and a whole-game match hit 0 of 854 decisions. Here every hand plays every preflop
node of a fixed menu, at a fixed set of stacks:

  isomorphism   information sets use the hand's class (AKs, AKo, 88, ...), not its suits: 1326 hands -> 169
  leaves        a flop leaf is valued by preflop/leafvalue.py (a model fitted to legacy rollouts), and a preflop
                all-in by the class-vs-class equity table, instead of a 25-40ms rollout or a sampled runout
  serving       the live stack maps to the nearest grid stack (within MAX_STACK_RATIO); an all-in maps to the grid
                all-in. Other sizes stay exact: a history with a size that is not on the menu is unseen, and the
                live engine falls back to its own preflop logic. Served all-ins and calls are mapped back to the
                live stack.

Readiness: every node checkpoints its average strategy at each power-of-two update count (from CHECKPOINT_FROM) and
serves under the candidate's readiness rule (coarse stability by default), as coverage targets do.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from preflop.exactcoverage import action_groups
from preflop.exactgame import Config, State
from preflop.exacttrain import READINESS, Trainer, _metadata
from preflop.leafvalue import LeafModel, class_combos, hand_class

STACK_GRID = (10.0, 15.0, 20.0, 27.0, 35.0, 45.0, 57.0, 70.0, 85.0, 100.0)
MENU = {"chip_bb": 0.05, "fractions": (0.75, 1.5), "extra_sizes_bb": (2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0)}
MAX_STACK_RATIO = 1.2           # serve a live stack only within 20% of a grid stack (beyond the grid's ends: none)
CHECKPOINT_FROM = 64


def grid_config(stacks=STACK_GRID, exploration=0.6):
    return Config(stacks_bb=tuple(stacks), exploration=exploration, **MENU)


def legacy_fingerprint():
    """The legacy continuation the leaf model was fitted to: bot sources and policy files, the game rules and the
    rollout driver. Narrower than exactdepth.continuation_fingerprint, so editing training code elsewhere in preflop/
    does not orphan a fitted leaf model."""
    from bot.audit import ROOT, _fingerprint, policy_files
    from bot.engine import Engine
    from preflop.exactdepth import ENGINE
    sources = sorted((ROOT / "bot").rglob("*.py")) + [ROOT / "preflop/exactgame.py", ROOT / "preflop/exactdepth.py"]
    return {"source_sha256": _fingerprint(sources, ROOT),
            "policy_sha256": _fingerprint(policy_files(Engine(seed=0, **ENGINE)), ROOT), "engine": ENGINE}


def canonical_key(state):
    """The actor's information-set key with its hole cards replaced by its class representative (preflop only)."""
    if state.board:
        raise ValueError("hand-class keys are preflop only")
    obs = state.observation()
    obs["hole"] = sorted(class_combos(hand_class(state.holes[state.actor]))[0])
    return json.dumps(obs, sort_keys=True, separators=(",", ":"))


def grid_stack(stack_bb, grid):
    nearest = min(grid, key=lambda g: abs(math.log(stack_bb / g)))
    return nearest if abs(math.log(stack_bb / nearest)) <= math.log(MAX_STACK_RATIO) else None


def map_context(context, grid):
    """(grid context, live stack) or (None, reason) for a live preflop context."""
    if context.get("board"):
        return None, "postflop"
    live = float(context["effective_stack_bb"])
    stack = grid_stack(live, grid)
    if stack is None:
        return None, "stack off grid"
    history = []
    for street in context["history"]:
        mapped = []
        for player, kind, to in street:
            if kind in ("bet", "raise"):
                if to >= live - 1e-9:
                    to = stack                                   # an all-in is the grid all-in
                elif to >= stack:
                    return None, "size above grid stack"
            mapped.append([player, kind, to])
        history.append(mapped)
    return {**context, "effective_stack_bb": stack, "history": history}, live


def live_options(options, state, context, live):
    """Grid engine options with all-ins (and sizes at or above the live stack) and calls in live amounts."""
    high = max([1.0] + [to for _, kind, to in context["history"][0] if kind == "raise"])
    result = {}
    for label, (kind, to) in options.items():
        if label == f"to:{state.cap}" or (kind in ("bet", "raise") and to >= live - 1e-9):
            to = live
        elif label == "call":
            to = min(high, live)
        result[label] = (kind, to)
    return result


class _GridState(State):
    """State with legal action menus memoised per public node (stack, history), which fully determine them. The
    game rules are State's; only repeated computation is skipped (35% of training time when profiled)."""

    def actions(self):
        cache = self.config.__dict__.setdefault("_menu_cache", {})
        key = (self.stack, self.history, self.folded, self.street)
        menu = cache.get(key)
        if menu is None:
            menu = cache[key] = State.actions(self)
        return menu


class PreflopGameTrainer(Trainer):
    depth_limited = True

    def __init__(self, path, config=None, seed=0, leaf=None, readiness=None, **kwargs):
        super().__init__(path, config or (None if Path(path).exists() else grid_config()), seed=seed, **kwargs)
        try:
            meta = _metadata(self.db)
            if "preflop_game" not in meta:
                if meta["iterations"] or leaf is None:
                    raise ValueError("preflop-game training needs a new candidate and a leaf model")
                model = LeafModel(leaf)
                if model.record.get("legacy") != legacy_fingerprint():
                    raise ValueError("leaf model was fitted to a different legacy engine")
                self._set("preflop_game", {"stack_grid": list(self.config.stacks_bb), "isomorphism": "hand_class",
                                           "leaf": {"path": str(Path(leaf).resolve()), "sha256": model.sha256,
                                                    "kind": model.record["kind"]},
                                           "allin": "class_equity", "checkpoint_from": CHECKPOINT_FROM})
                self._set("continuation", model.record["legacy"])
                self._set("readiness", readiness or READINESS)
                self.db.execute("CREATE TABLE target_checkpoints (key TEXT PRIMARY KEY, updates INTEGER NOT NULL, "
                                "probs TEXT NOT NULL, stability REAL)")
                self.db.commit()
            meta = _metadata(self.db)
            record = meta["preflop_game"]
            self.leaf = LeafModel(leaf or record["leaf"]["path"])
            if self.leaf.sha256 != record["leaf"]["sha256"]:
                raise ValueError("leaf model file changed")
            if meta["continuation"] != legacy_fingerprint():
                raise ValueError("legacy engine changed since the leaf model was fitted; refit it")
            self.readiness = meta["readiness"]
        except BaseException:
            self.close()
            raise

    def _root(self):
        return _GridState.deal(self.config, self.rng)

    def _walk(self, state, updater, own=1.0, opponent=1.0, sampled=1.0):
        if state.terminal:
            if state.folded is not None:
                return state.utility(updater)
            # Preflop all-in: class-vs-class equity instead of the deal's runout (lower variance, class-level bias).
            equity = self.leaf.equity[hand_class(state.holes[0]), hand_class(state.holes[1])]
            btn = (equity * state.pot - state.spent[0]) * state.config.chip_bb
            return float(btn if updater == 0 else -btn)
        if state.street >= 1:
            return self.leaf.value(state, updater)
        return super()._walk(state, updater, own, opponent, sampled)

    def _node(self, state):
        key, actions = canonical_key(state), state.actions()
        row = self.store.get(key)
        if row and row[0] != actions:
            raise ValueError("legal action schema changed for an existing information set")
        positive = np.maximum(row[1], 0) if row else np.zeros(len(actions))
        strategy = positive / positive.sum() if positive.sum() > 0 else np.full(len(actions), 1 / len(actions))
        return key, actions, row, strategy

    def _record(self, state, key, actions, row, delta, average, updating):
        super()._record(state, key, actions, row, delta, average, updating)
        n = (row[4] if row else 0) + 1
        if not updating or n < CHECKPOINT_FROM or n & (n - 1):
            return
        sums = self.store.get(key)[2]
        if not sums.sum() > 0:
            return
        probs = sums / sums.sum()
        cp = self.db.execute("SELECT probs FROM target_checkpoints WHERE key=?", (key,)).fetchone()
        if cp is None:
            self.db.execute("INSERT INTO target_checkpoints VALUES (?,?,?,NULL)", (key, n, json.dumps(probs.tolist())))
            return
        old = np.asarray(json.loads(cp[0]))
        if self.readiness.get("stability", "full") == "coarse":
            groups = action_groups(actions, state.cap)
            tv = 0.5 * np.abs(np.bincount(groups, probs, 4) - np.bincount(groups, old, 4)).sum()
        else:
            tv = 0.5 * np.abs(probs - old).sum()
        self.db.execute("UPDATE target_checkpoints SET updates=?,probs=?,stability=? WHERE key=?",
                        (n, json.dumps(probs.tolist()), float(tv), key))

    def coverage(self, minimum=1000):
        """Ready nodes (update floor and stable) among nodes with a checkpoint, by stack."""
        from preflop.exacttrain import required_updates, stable
        ready, by_stack = 0, {}
        for (key,) in self.db.execute("SELECT key FROM target_checkpoints"):
            row = self.store.get(key)
            stack = str(json.loads(key)["stack"] * self.config.chip_bb)
            ok = row is not None and row[4] >= required_updates(len(row[0]), minimum, self.readiness) and \
                stable(self.db, key, self.readiness)
            entry = by_stack.setdefault(stack, {"checkpointed": 0, "ready": 0})
            entry["checkpointed"] += 1
            entry["ready"] += int(ok)
            ready += int(ok)
        return {"ready_nodes": ready, "by_stack_bb": by_stack, "readiness": self.readiness,
                "scope": "preflop_game_grid", "live_eligible": False,
                "reason": "requires_frozen_evaluation_and_promotion"}


def stamp_leaf_model(path):
    """Record the legacy fingerprint in a fitted leaf_model.json (once, right after fitting)."""
    path = Path(path)
    record = json.loads(path.read_text())
    record["legacy"] = legacy_fingerprint()
    path.write_text(json.dumps(record))
    return hashlib.sha256(path.read_bytes()).hexdigest()
