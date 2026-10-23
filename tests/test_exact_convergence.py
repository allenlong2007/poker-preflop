"""The same sampler must reduce exploitability in a tractable imperfect-information game."""
import itertools
import json
from dataclasses import dataclass

import numpy as np

from preflop.exactgame import Config
from preflop.exacttrain import Trainer


@dataclass(frozen=True)
class Kuhn:
    holes: tuple
    history: str = ""
    street: int = 0

    @property
    def actor(self):
        return len(self.history) % 2

    @property
    def terminal(self):
        return self.history in ("cc", "bf", "bc", "cbf", "cbc")

    def key(self):
        return f"{self.actor}:{self.holes[self.actor]}:{self.history}"

    def actions(self):
        return ("f", "c") if self.history in ("b", "cb") else ("c", "b")

    def step(self, action):
        return Kuhn(self.holes, self.history + action)

    def utility(self, player):
        if self.history == "bf":
            result = 1
        elif self.history == "cbf":
            result = -1
        else:
            result = (1 if self.holes[0] > self.holes[1] else -1) * (1 if self.history == "cc" else 2)
        return float(result if player == 0 else -result)


def test_sampled_mccfr_reduces_kuhn_exploitability(tmp_path, monkeypatch):
    with Trainer(tmp_path / "kuhn.sqlite", Config(chip_bb=0.5, stacks_bb=(3,)), seed=211) as trainer:
        monkeypatch.setattr(trainer, "_root", lambda: Kuhn(tuple(trainer.rng.choice(3, 2, replace=False))))
        trainer.train(10000)
        policy = {}
        for key, total in trainer.db.execute("SELECT key,total FROM nodes"):
            values = np.array(json.loads(total))
            policy[key] = values / values.sum() if values.sum() else np.array([0.5, 0.5])

    def value(state, br_player, plan):
        if state.terminal:
            return state.utility(0)
        key = state.key()
        if state.actor == br_player:
            probabilities = np.eye(2)[plan[key]]
        else:
            probabilities = policy[key]
        return sum(p * value(state.step(a), br_player, plan) for a, p in zip(state.actions(), probabilities))

    best = []
    for player in (0, 1):
        keys = sorted(k for k in policy if k.startswith(str(player) + ":"))
        values = []
        for choices in itertools.product((0, 1), repeat=len(keys)):
            plan = dict(zip(keys, choices))
            values.append(np.mean([value(Kuhn(deal), player, plan) for deal in itertools.permutations(range(3), 2)]))
        best.append(max(values) if player == 0 else min(values))
    exploitability = best[0] - best[1]
    assert exploitability < 0.12, exploitability
