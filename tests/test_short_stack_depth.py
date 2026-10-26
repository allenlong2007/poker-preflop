"""Experiment 49: stack-depth-consistent ranges and low-SPR play."""
import json

import numpy as np
import pytest

from bot.depth import BlendedNodes, bracket
from bot.engine import POLICY, Act, Engine
from bot import potodds
from preflop.cards import HANDS, n_combos

PRE = json.load(open(POLICY / "preflop.json"))
COMBOS = np.array([n_combos(h) for h in HANDS], float)


def share(weights):
    w = (weights - potodds.FLOOR * 0.25) / (1 - potodds.FLOOR)
    return float((w * COMBOS).sum() / COMBOS.sum())


def test_blend_equals_each_solved_tree_and_rows_sum_to_one():
    for depth, tree in ((10.0, PRE["by_depth"]["10"]), (100.0, PRE["nodes"])):
        n = BlendedNodes(PRE, depth)
        assert n.exact
        for a, v in tree["btn_root"].items():
            assert np.allclose(n["btn_root"][a], v)
    lo, hi, w = bracket(PRE, 30)
    assert (lo, hi) == (25.0, 40.0) and 0 < w < 1
    node = BlendedNodes(PRE, 30)["bb_vs_raise"]
    assert np.allclose(sum(node.values()), 1.0)
    assert bracket(PRE, 30, nearest=True) == (25.0, 25.0, 1.0)


def test_opponent_ranges_follow_the_effective_stack():
    shove = lambda s: share(potodds.preflop_weights(PRE, [Act("btn", "raise", s)], "btn", s))
    opens = lambda s: share(potodds.preflop_weights(PRE, [Act("btn", "raise", 2.25)], "btn", s))
    assert shove(6) > shove(10) > shove(20)                # shorter stacks shove wider
    assert opens(100) > opens(15) > opens(6)               # and min-raise less (they shove instead)
    # A shove where the depth tree never shoves (60bb) is read from the nearest depth that does, not as any hand.
    assert shove(60) < 0.2
    # The all-in test is relative to the stack: 9bb at 10bb deep is a shove, at 100bb it is a raise.
    assert potodds._node([], Act("btn", "raise", 9), "btn", 10)[1] == "all-in"
    assert potodds._node([], Act("btn", "raise", 9), "btn", 100)[1] == "raise"


def test_engine_plays_and_reads_from_the_same_depth():
    eng = Engine(exact_required=False, seed=0)
    d = eng.decide("btn", "Kh9c", "", [[]], sample=False, stack=12)
    assert "blend" in d.node and d.diagnostics["lookup"]["depth_blend"]["depth_lo"] == 10.0
    rates = [eng._tree_shove_rate("btn_first_in", s) for s in (100, 20, 10, 6)]
    assert rates == sorted(rates) and rates[0] < 0.01 < 0.25 < rates[2]
    old = Engine(exact_required=False, seed=0)
    old.depth_blend = False
    assert "(10bb tree)" in old.decide("btn", "Kh9c", "", [[]], sample=False, stack=12).node


def test_self_play_agents_pass_the_real_stack():
    from bot.selfplay import DECK, EngineAgent, play_hand
    agent = EngineAgent(Engine(exact_required=False, seed=1), stack=10)
    other = EngineAgent(Engine(exact_required=False, seed=2), stack=10)
    rng = np.random.default_rng(0)
    for _ in range(30):
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        play_hand({"btn": agent, "bb": other}, {"btn": c[:2], "bb": c[2:4]}, c[4:], stack=10)
    assert agent.sources.get("short", 0) > 0              # low-SPR mode is reached at 10bb


@pytest.mark.parametrize("stack", [8, 12])
def test_low_spr_ev_mixes_only_near_equal_options(stack):
    eng = Engine(exact_required=False, seed=0)
    eng.low_spr = "ev"
    hist = [[Act("btn", "raise", 2.25), Act("bb", "call")], [Act("bb", "check")]]
    d = eng.decide("btn", "AhKd", "Kh8d3c", hist, sample=False, stack=stack)
    ev = d.diagnostics["short_stack"]["ev"]
    best = max(ev, key=ev.get)
    pot = 4.5
    for k, p in d.probs.items():
        if ev[k] < ev[best] - 0.1 * pot:
            assert p < 0.01                                 # clearly worse options are (almost) never chosen
