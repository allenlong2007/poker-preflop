"""Durable tabular outcome-sampling MCCFR for exact-card no-limit hold'em.

One sampled trajectory per updating player, signed importance-weighted regrets, and
reach-weighted average strategies. No neural generalization or convergence claim.
"""
import fcntl
import hashlib
import json
import math
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np

from preflop.exactgame import Config, State, from_context

SCHEMA = 1


def outcome_update(policy, chosen, probability, child, own, opponent, sampled):
    """Unbiased sampled action values; compensate for exploration and prefix reach."""
    if probability <= 0 or sampled <= 0:
        raise FloatingPointError("sampling reach underflow")
    action_values = np.zeros(len(policy))
    action_values[chosen] = child / probability
    value = float(policy @ action_values)
    regret = (opponent / sampled) * (action_values - value)
    average = (own / sampled) * policy
    if not np.isfinite(regret).all() or not np.isfinite(average).all() or not math.isfinite(value):
        raise FloatingPointError("non-finite importance weighted update; batch rolled back")
    return regret, average, value


def _code_hash():
    paths = (Path(__file__), Path(__file__).with_name("exactgame.py"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def _metadata(db):
    return {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}


def _public_focus(focus):
    roots = []
    for context in focus or []:
        root = {k: context[k] for k in ("hero_position", "board", "effective_stack_bb", "history")}
        if root not in roots:
            roots.append(root)
    return roots


class Trainer:
    """Single writer; each batch commits nodes, iteration, and RNG state together."""

    def __init__(self, path, config=None, seed=0, focus=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lease = self.path.with_suffix(self.path.suffix + ".lock").open("a")
        self.db = None
        try:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(self.path, timeout=30)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS nodes (key TEXT PRIMARY KEY, actions TEXT NOT NULL, "
                            "regrets TEXT NOT NULL, total TEXT NOT NULL, visits INTEGER NOT NULL, "
                            "updates INTEGER NOT NULL, street INTEGER NOT NULL)")
            meta = _metadata(self.db)
            if not meta:
                self.config = config or Config()
                self.focus = _public_focus(focus)
                self.rng = np.random.default_rng(seed)
                for root in self.focus:
                    from_context(self.config, root, np.random.default_rng(0))
                self._set("schema", SCHEMA)
                self._set("algorithm", "outcome_sampling_mccfr")
                self._set("config", asdict(self.config))
                self._set("config_sha256", self.config.fingerprint())
                self._set("training_code_sha256", _code_hash())
                self._set("focus", self.focus)
                self._set("focus_prior", "uniform_compatible_private_combos")
                self._set("seed", seed)
                self._set("iterations", 0)
                self._set("rng", self.rng.bit_generator.state)
                self.db.commit()
            else:
                if meta.get("schema") != SCHEMA or meta.get("algorithm") != "outcome_sampling_mccfr":
                    raise ValueError("incompatible training schema")
                self.config = Config(**meta["config"])
                if config is not None and config.fingerprint() != self.config.fingerprint():
                    raise ValueError("resume configuration mismatch")
                if meta.get("training_code_sha256") != _code_hash():
                    raise ValueError("training code changed; start a new candidate rather than mixing algorithms")
                self.focus = meta["focus"]
                if focus is not None and _public_focus(focus) != self.focus:
                    raise ValueError("resume focus dataset mismatch")
                if meta.get("promotion"):
                    raise ValueError("cannot train a promoted serving artifact")
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = meta["rng"]
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if not self._lease.closed:
            self._lease.close()

    def _set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                        (key, json.dumps(value, allow_nan=False, sort_keys=True)))

    def _root(self):
        if self.focus and self.rng.random() < 0.8:
            context = self.focus[int(self.rng.integers(len(self.focus)))]
            return from_context(self.config, context, self.rng)
        return State.deal(self.config, self.rng)

    def _walk(self, state, updater, own=1.0, opponent=1.0, sampled=1.0):
        if state.terminal:
            return state.utility(updater)
        key, actions = state.key(), state.actions()
        row = self.db.execute("SELECT actions,regrets,total,visits,updates FROM nodes WHERE key=?", (key,)).fetchone()
        if row:
            if tuple(json.loads(row[0])) != actions:
                raise ValueError("legal action schema changed for an existing information set")
            regrets, total = np.array(json.loads(row[1])), np.array(json.loads(row[2]))
            visits, updates = row[3:]
        else:
            regrets, total, visits, updates = np.zeros(len(actions)), np.zeros(len(actions)), 0, 0
        positive = np.maximum(regrets, 0)
        strategy = positive / positive.sum() if positive.sum() > 0 else np.full(len(actions), 1 / len(actions))
        updating = state.actor == updater
        behavior = (1 - self.config.exploration) * strategy + self.config.exploration / len(actions) \
            if updating else strategy
        chosen = int(self.rng.choice(len(actions), p=behavior))
        child = self._walk(state.step(actions[chosen]), updater,
                           own * strategy[chosen] if updating else own,
                           opponent if updating else opponent * strategy[chosen], sampled * behavior[chosen])
        delta, average, value = outcome_update(strategy, chosen, behavior[chosen], child, own, opponent, sampled)
        if updating:
            regrets += delta
            total += average
            updates += 1
        self.db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?)",
                        (key, json.dumps(actions), json.dumps(regrets.tolist(), allow_nan=False),
                         json.dumps(total.tolist(), allow_nan=False), visits + 1, updates, state.street))
        return value

    def train(self, iterations):
        if iterations <= 0:
            raise ValueError("positive iteration count required")
        original = json.dumps(self.rng.bit_generator.state)
        try:
            self.db.execute("BEGIN IMMEDIATE")
            for _ in range(iterations):
                for player in (0, 1):
                    self._walk(self._root(), player)
            completed = _metadata(self.db)["iterations"] + iterations
            self._set("iterations", completed)
            self._set("rng", self.rng.bit_generator.state)
            self.db.commit()
        except BaseException:
            self.db.rollback()
            self.rng.bit_generator.state = json.loads(original)
            raise

    def status(self):
        meta = _metadata(self.db)
        count, updates = self.db.execute("SELECT count(*),coalesce(sum(updates),0) FROM nodes").fetchone()
        streets = {str(s): {"infosets": n, "updates": u}
                   for s, n, u in self.db.execute("SELECT street,count(*),sum(updates) FROM nodes GROUP BY street")}
        return {"iterations": meta["iterations"], "infosets": count, "updates": updates,
                "streets": streets, "config_sha256": meta["config_sha256"], "focus_roots": len(self.focus)}

    def checkpoint(self, path):
        """SQLite backup produces a self-contained immutable evaluation snapshot (includes WAL)."""
        destination = Path(path)
        if destination.resolve() == self.path.resolve():
            raise ValueError("checkpoint must differ from the writable candidate")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with sqlite3.connect(temporary) as target:
            self.db.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")
        os.replace(temporary, destination)


class ExactPolicy:
    """Read-only average strategy. Unseen/undertrained/unpromoted states never use uniform play."""

    def __init__(self, path, min_updates=1000, candidate=False):
        if min_updates < 1:
            raise ValueError("min_updates must be positive")
        self.path = Path(path)
        self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        try:
            self.meta = _metadata(self.db)
            if self.meta.get("schema") != SCHEMA:
                raise ValueError("incompatible exact policy schema")
            self.config = Config(**self.meta["config"])
            self.minimum, self.candidate = min_updates, candidate
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.db.close()

    def lookup(self, context):
        info = {"iterations": self.meta["iterations"], "config_sha256": self.meta["config_sha256"]}
        if not self.candidate and not self.meta.get("promotion"):
            return {**info, "status": "unpromoted"}
        try:
            state = from_context(self.config, context, np.random.default_rng(0), known_hole=True)
        except (ValueError, KeyError, TypeError) as error:
            return {**info, "status": "unsupported", "reason": str(error)}
        row = self.db.execute("SELECT actions,total,visits,updates FROM nodes WHERE key=?", (state.key(),)).fetchone()
        if not row:
            return {**info, "status": "unseen"}
        actions, sums = tuple(json.loads(row[0])), np.array(json.loads(row[1]))
        info.update({"visits": row[2], "updates": row[3]})
        if actions != state.actions() or not np.isfinite(sums).all() or (sums < 0).any():
            return {**info, "status": "invalid"}
        if row[3] < self.minimum or sums.sum() <= 0:
            return {**info, "status": "insufficient"}
        return {**info, "status": "ready", "probs": dict(zip(actions, map(float, sums / sums.sum()))),
                "options": state.engine_options()}


def promotion_errors(report):
    """Predeclared minimum evidence, not a claim of equilibrium or universal strength."""
    required = ("deals", "coverage", "bb100", "se_bb100", "illegal")
    if any(k not in report or not math.isfinite(report[k]) for k in required):
        return ["missing or non-finite evaluation metrics"]
    errors = []
    if report["deals"] < 2000:
        errors.append("need at least 2000 held-out duplicate deals")
    if not 0.25 <= report["coverage"] <= 1:
        errors.append("exact decision coverage must be at least 25%")
    if report["se_bb100"] < 0 or report["bb100"] - 1.96 * report["se_bb100"] < -5:
        errors.append("95% lower bound must exceed -5 bb/100 against baseline")
    if report["illegal"] != 0:
        errors.append("illegal actions observed")
    return errors
