"""scripts/live_context_eval.py: paired plays from live contexts against the frozen baseline."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_context_eval as lce  # noqa: E402
from preflop.exactgame import Config, from_context  # noqa: E402

CONFIG = {"chip_bb": 0.01}
FLOP = {"hero_position": "btn", "hero_hole": "AsKd", "board": "Kh8d3c", "effective_stack_bb": 100.0,
        "history": [[["btn", "raise", 2.5], ["bb", "call", 0.0]], [["bb", "check", 0.0]], [], []]}
OPEN = {"hero_position": "btn", "hero_hole": "7sAd", "board": "", "effective_stack_bb": 97.0,
        "history": [[], [], [], []]}


def _setup(monkeypatch):
    # No exact artifact: the "candidate" is the legacy engine, so every paired difference must be exactly zero.
    from bot.engine import Engine
    engines = tuple(Engine(exact_required=False, seed=0, pot_odds=True, exact_policy_path=False) for _ in range(3))
    monkeypatch.setattr(lce, "_W", {"engines": engines, "config": Config(**CONFIG)})


def test_identical_policies_score_exactly_zero(monkeypatch):
    _setup(monkeypatch)
    for context in (FLOP, OPEN):
        _, res = lce.run_target((0, context, 6, 30, 11))
        assert len(res["uniform"]) == 6 and len(res["posterior"]) == 6
        assert all(r[0] == 0 for r in res["uniform"] + res["posterior"])
        assert res["illegal"] == 0 and all(r[1] == 0 for r in res["uniform"])     # nothing served exact


def test_posterior_weights_condition_on_the_villain_line(monkeypatch):
    _setup(monkeypatch)
    config, engines = Config(**CONFIG), lce._W["engines"]
    root = from_context(config, OPEN, np.random.default_rng(0), known_hole=True)
    assert lce._weight(root, engines[2], OPEN, engines, 0) == 1.0               # no villain action yet
    weights = [lce._weight(from_context(config, FLOP, np.random.default_rng(k), known_hole=True), engines[2], FLOP,
                           engines, k) for k in range(40)]
    assert all(0 <= w <= 1 for w in weights) and len({round(w, 6) for w in weights}) > 1


def test_runs_are_reproducible(monkeypatch):
    _setup(monkeypatch)
    a = lce.run_target((3, FLOP, 5, 20, 7))
    b = lce.run_target((3, FLOP, 5, 20, 7))
    assert a == b


def test_diagnosis_maps_actions_to_legal_game_labels():
    import live_context_diagnose as lcd
    state = from_context(Config(**CONFIG), FLOP, np.random.default_rng(0), known_hole=True)
    assert lcd._label(state, "check", 0) == "check"                 # nothing to call: a call is a check
    assert lcd._label(state, "fold", 0) == "fold"
    big = lcd._label(state, "bet", 1e6)
    assert big == f"to:{state.cap}" and big in state.actions()      # oversized bets clip to all-in
    small = lcd._label(state, "bet", 0.01)
    assert int(small[3:]) == state.minimum
