"""Exact information boundaries and heads-up no-limit chip accounting."""
import json

import numpy as np
import pytest

from preflop.exactgame import Config, State, from_context


def test_information_set_keeps_suits_kickers_history_but_not_hidden_cards():
    cfg = Config(chip_bb=0.5)
    a = State.deal(cfg, np.random.default_rng(1), hole=("As", "Kd"), board=("Kh", "8d", "3c", "2s", "9h"))
    b = State.deal(cfg, np.random.default_rng(2), hole=("As", "Kd"), board=("Kh", "8d", "3c", "4s", "Th"))
    assert a.key() == b.key()  # no opponent cards or future board before it is revealed
    a = a.step("call").step("check")
    b = b.step("call").step("check")
    assert a.actor == 1 and a.street == 1
    # BB private cards differ; ask for the BTN's observation explicitly.
    assert a.key(player=0) == b.key(player=0)
    c = State.deal(cfg, np.random.default_rng(1), hole=("Ah", "Kd"), board=("Ks", "8d", "3c", "2s", "9h"))
    c = c.step("call").step("check")
    assert c.key(player=0) != a.key(player=0)
    assert "9h" not in json.loads(a.key(player=0))["board"]


def test_blinds_limp_checks_minraise_and_showdown_are_zero_sum():
    cfg = Config(chip_bb=0.5)
    s = State.deal(cfg, np.random.default_rng(3), stack_bb=10)
    assert s.pot == 3 and s.facing == 1 and "to:3" not in s.actions()
    assert "to:4" in s.actions()
    s = s.step("call")
    assert s.street == 0 and s.actor == 1 and s.facing == 0
    s = s.step("check")
    assert s.street == 1 and s.actor == 1 and s.pot == 4
    for _ in range(6):
        s = s.step("check")
    assert s.terminal
    assert s.utility(0) == -s.utility(1)


def test_fold_allin_and_short_raise_are_legal():
    s = State.deal(Config(chip_bb=0.5), np.random.default_rng(1), stack_bb=2.5)
    folded = s.step("fold")
    assert folded.utility(0) == -0.5 and folded.utility(1) == 0.5
    s = s.step("to:4")  # open to 2bb, leaving 0.5bb
    assert "to:5" in s.actions()  # short all-in raise below normal minimum
    s = s.step("to:5")
    assert set(s.actions()) == {"fold", "call"}
    assert s.step("call").terminal
    with pytest.raises(ValueError):
        s.step("check")


def test_all_chip_sizes_and_observed_off_menu_sizes_are_preserved():
    cfg = Config(chip_bb=0.5, exhaustive=True)
    s = State.deal(cfg, np.random.default_rng(1), stack_bb=5)
    assert {f"to:{n}" for n in range(4, 11)} <= set(s.actions())
    context = {"hero_position": "bb", "hero_hole": "AsKh", "board": "",
               "effective_stack_bb": 5, "history": [[["btn", "raise", 2.5]]]}
    live = from_context(cfg, context, np.random.default_rng(8), known_hole=True)
    assert live.facing == 3 and live.history[0][0] == (0, "raise", 5)
    with pytest.raises(ValueError, match="precision"):
        from_context(cfg, {**context, "history": [[["btn", "raise", 2.51]]]}, np.random.default_rng(8))


def test_random_play_never_loses_chips_or_creates_illegal_actions():
    rng = np.random.default_rng(77)
    cfg = Config(chip_bb=0.5)
    for _ in range(80):
        s = State.deal(cfg, rng, stack_bb=6)
        for _ in range(100):
            if s.terminal:
                break
            choices = s.actions()
            s = s.step(choices[int(rng.integers(len(choices)))])
            assert all(0 <= x <= s.stack for x in s.spent)
        assert s.terminal
        assert abs(s.utility(0)) <= 6
        assert s.utility(0) + s.utility(1) == 0
