"""Heads-up no-limit game with exact cards, perfect recall, and integer chip accounting.

No textures, strength buckets, opponent cards, or future runouts enter information keys.
Default action menus are configurable abstractions; exhaustive mode uses every legal chip total.
"""
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from decimal import Decimal

import eval7
import numpy as np

DECK = tuple(r + s for r in "23456789TJQKA" for s in "cdhs")
SEATS = ("btn", "bb")


def cards(value):
    if isinstance(value, str):
        value = value.replace(" ", "")
        value = tuple(value[i:i + 2] for i in range(0, len(value), 2))
    result = tuple(map(str, value))
    if any(card not in DECK for card in result) or len(set(result)) != len(result):
        raise ValueError("invalid or duplicate cards")
    return result


@dataclass(frozen=True)
class Config:
    chip_bb: float = 0.01
    fractions: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    stacks_bb: tuple = (6.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0)
    exhaustive: bool = False
    exploration: float = 0.6
    extra_sizes_bb: tuple = ()

    def __post_init__(self):
        if not math.isfinite(self.chip_bb) or self.chip_bb <= 0 or self.chip_bb > 0.5:
            raise ValueError("chip_bb must be in (0, 0.5]")
        if not 0 < self.exploration <= 1:
            raise ValueError("exploration must be in (0, 1]")
        object.__setattr__(self, "fractions", tuple(sorted(set(self.fractions))))
        object.__setattr__(self, "stacks_bb", tuple(self.stacks_bb))
        if not self.fractions or any(not math.isfinite(f) or f <= 0 for f in self.fractions):
            raise ValueError("positive finite sizing fractions required")
        if not self.stacks_bb or any(not math.isfinite(s) or s < 1 for s in self.stacks_bb):
            raise ValueError("stacks must be at least one big blind")
        object.__setattr__(self, "extra_sizes_bb", tuple(sorted(set(self.extra_sizes_bb))))
        if any(not math.isfinite(v) or v <= 0 for v in self.extra_sizes_bb):
            raise ValueError("extra sizes must be positive finite chip totals")
        for amount in self.extra_sizes_bb:
            self.chips(amount)
        self.chips(0.5)
        for stack in self.stacks_bb:
            self.chips(stack)

    def chips(self, bb):
        value = float(Decimal(str(bb)) / Decimal(str(self.chip_bb)))
        if not math.isfinite(value) or abs(value - round(value)) > 1e-5:
            raise ValueError("amount exceeds configured chip precision")
        return round(value)

    def fingerprint(self):
        values = asdict(self)
        if not self.extra_sizes_bb:
            values.pop("extra_sizes_bb")  # retain existing immutable artifact fingerprints
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class State:
    config: Config
    holes: tuple
    runout: tuple
    stack: int
    street: int = 0
    actor: int = 0
    spent: tuple = (0, 0)
    put: tuple = (0, 0)
    history: tuple = ((), (), (), ())
    last_raise: int = 0
    folded: int | None = None
    root_id: str = "whole_game"

    @classmethod
    def deal(cls, config, rng, stack_bb=None, hole=None, board=()):
        known, board = cards(hole or ()), cards(board)
        if len(known) not in (0, 2) or len(board) not in (0, 3, 4, 5) or set(known) & set(board):
            raise ValueError("invalid private/public card deal")
        available = [c for c in DECK if c not in set(known) | set(board)]
        drawn = list(rng.choice(available, size=9 - len(known) - len(board), replace=False))
        first = known or tuple(drawn[:2])
        if not known:
            drawn = drawn[2:]
        second, tail = tuple(drawn[:2]), tuple(drawn[2:])
        stack_bb = stack_bb if stack_bb is not None else config.stacks_bb[int(rng.integers(len(config.stacks_bb)))]
        stack = config.chips(stack_bb)
        if stack < config.chips(1):
            raise ValueError("effective stack must cover the big blind")
        blinds = (config.chips(0.5), config.chips(1))
        return cls(config, (tuple(first), second), board + tail, stack, spent=blinds, put=blinds,
                   last_raise=blinds[1])

    @property
    def terminal(self):
        return self.folded is not None or self.street == 4

    @property
    def board(self):
        visible = self.runout[:(0, 3, 4, 5, 5)[self.street]]
        return tuple(sorted(visible[:3], key=DECK.index)) + visible[3:]

    @property
    def pot(self):
        return sum(self.spent)

    @property
    def facing(self):
        return max(self.put) - self.put[self.actor]

    @property
    def cap(self):
        return self.stack - self.spent[self.actor] + self.put[self.actor]

    @property
    def minimum(self):
        return max(self.put) + self.last_raise if max(self.put) else self.config.chips(1)

    def actions(self):
        if self.terminal:
            return ()
        basic = ["fold", "call"] if self.facing else ["check"]
        other = 1 - self.actor
        high = max(self.put)
        if self.spent[other] >= self.stack or self.cap <= high:
            return tuple(basic)
        if self.config.exhaustive:
            sizes = set(range(self.minimum, self.cap + 1))
        else:
            # Raises are increments over the call, as a fraction of the pot after calling.
            sizes = {high + round(f * (self.pot + self.facing)) for f in self.config.fractions}
            sizes.update(self.config.chips(v) for v in self.config.extra_sizes_bb)
            sizes.add(self.minimum)
        sizes = {size for size in sizes if self.minimum <= size <= self.cap}
        sizes.add(self.cap)  # includes a short all-in raise, which need not meet the normal minimum
        return tuple(basic + [f"to:{size}" for size in sorted(sizes) if size > high])

    def observation(self, player=None):
        player = self.actor if player is None else player
        return {"schema": 1, "root_id": self.root_id, "hole": sorted(self.holes[player]), "board": list(self.board),
                "player": player, "actor": self.actor, "street": self.street, "stack": self.stack,
                "spent": self.spent, "put": self.put, "last_raise": self.last_raise,
                "chip_bb": self.config.chip_bb, "history": self.history}

    def key(self, player=None):
        return json.dumps(self.observation(player), sort_keys=True, separators=(",", ":"))

    def public_root_id(self):
        public = self.observation()
        public.pop("hole")
        public.pop("root_id")
        return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()

    def step(self, action, observed=False):
        if self.terminal:
            raise ValueError("terminal state")
        if action not in self.actions():
            # Histories may contain legal sizes outside the training menu; preserve them exactly.
            size = int(action[3:]) if action.startswith("to:") else -1
            if not (observed and size > max(self.put) and size <= self.cap and
                    (size >= self.minimum or size == self.cap) and self.spent[1 - self.actor] < self.stack):
                raise ValueError(f"illegal action {action}")
        actor, other = self.actor, 1 - self.actor
        put, spent = list(self.put), list(self.spent)
        last_raise = self.last_raise
        if action == "fold":
            kind, amount = "fold", 0
        elif action == "check":
            kind, amount = "check", 0
        elif action == "call":
            kind, amount = "call", 0
            target = min(max(put), self.cap)
            spent[actor] += target - put[actor]
            put[actor] = target
        else:
            target = int(action[3:])
            kind, amount = ("raise" if self.street == 0 or max(put) else "bet"), target
            increment = target - max(put)
            last_raise = max(last_raise, increment)
            spent[actor] += target - put[actor]
            put[actor] = target
        histories = list(self.history)
        previous = histories[self.street]
        histories[self.street] = previous + ((actor, kind, amount),)
        state = replace(self, spent=tuple(spent), put=tuple(put), history=tuple(histories),
                        actor=other, last_raise=last_raise)
        if action == "fold":
            return replace(state, folded=actor)
        closes = (action == "call" and not (self.street == 0 and not previous)) or \
                 (action == "check" and (self.street == 0 or bool(previous)))
        if closes:
            street = 4 if max(spent) == self.stack else self.street + 1
            state = replace(state, street=street, actor=1, put=(0, 0), last_raise=self.config.chips(1))
        return state

    def utility(self, player):
        if not self.terminal:
            raise ValueError("utility is only available at terminal states")
        if self.folded is not None:
            btn = -self.spent[0] if self.folded == 0 else self.spent[1]
        else:
            board = [eval7.Card(c) for c in self.runout]
            scores = [eval7.evaluate([eval7.Card(c) for c in hand] + board) for hand in self.holes]
            share = 1 if scores[0] > scores[1] else 0.5 if scores[0] == scores[1] else 0
            btn = share * self.pot - self.spent[0]
        return float(btn * self.config.chip_bb * (1 if player == 0 else -1))

    def engine_options(self):
        result = {}
        for label in self.actions():
            if label.startswith("to:"):
                kind = "raise" if self.street == 0 or max(self.put) else "bet"
                result[label] = (kind, int(label[3:]) * self.config.chip_bb)
            else:
                to = min(max(self.put), self.cap) * self.config.chip_bb if label == "call" else 0.0
                result[label] = (label, to)
        return result


def from_context(config, context, rng, known_hole=False):
    """Reconstruct a public subgame. Training samples BOTH private hands, ignoring revealed labels.

    Uniform compatible ranges are the default subgame prior. They are not a solved posterior
    for an arbitrary observed betting line. Inference may supply the acting player's known cards.
    """
    player = SEATS.index(context["hero_position"])
    hole = context.get("hero_hole") if known_hole else None
    state = State.deal(config, rng, stack_bb=context["effective_stack_bb"], hole=hole,
                       board=context.get("board", ""))
    if known_hole and player == 1:
        state = replace(state, holes=(state.holes[1], state.holes[0]))
    for street, actions in enumerate(context["history"]):
        for p, kind, to in actions:
            if state.terminal or state.street != street or state.actor != SEATS.index(p):
                raise ValueError("inconsistent public action history")
            label = f"to:{config.chips(to)}" if kind in ("bet", "raise") else kind
            state = state.step(label, observed=True)
    if state.terminal or len(state.board) != len(cards(context.get("board", ""))):
        raise ValueError("context is terminal or board does not match its betting street")
    if known_hole and state.actor != player:
        raise ValueError("hero is not the next actor")
    return state
