"""Checks for the match bot's engine and dealer (experiment 29)."""
import numpy as np
import pytest

from bot.engine import POLICY, Act, Engine, State, map_combo, parse_cards
from bot.selfplay import Maniac, Station, EngineAgent, match, play_hand
from preflop import fullgame as fg
from preflop.flopgame import texture

pytestmark = pytest.mark.skipif(not (POLICY / "preflop.json").exists(), reason="match policies not built")


def test_state_pot_and_street():
    st = State([[Act("btn", "raise", 3), Act("bb", "call")], [Act("bb", "check"), Act("btn", "bet", 2)]])
    assert st.street == 1 and st.street_start[1] == (6.0, 97.0)
    assert st.facing("bb") == 2 and st.pot() == 8
    assert State([[Act("btn", "call")]]).street == 0            # the limp leaves the BB its option
    assert State([[Act("btn", "call"), Act("bb", "check")]]).street == 1


def test_map_combo_same_flop_is_identity():
    flop = parse_cards("Kh8d3c")
    hole = parse_cards("AcQd")
    assert map_combo(hole, flop, flop) == hole


def test_map_combo_keeps_hand_group_and_class():
    rng = np.random.default_rng(0)
    deck = parse_cards("".join(r + s for r in "23456789TJQKA" for s in "cdhs"))
    eng = Engine()
    same, tried = 0, 0
    for _ in range(300):
        cards = [deck[i] for i in rng.choice(52, 5, replace=False)]
        flop, hole = cards[:3], cards[3:]
        meta = eng.index.get(str(texture(flop)))
        if meta is None:
            continue
        rep = parse_cards(meta["flop"])
        assert texture(rep) == texture(flop)
        m = map_combo(hole, flop, rep)
        if m is not None:
            tried += 1
            same += fg.fine_bucket(m, rep) == fg.fine_bucket(hole, flop)
    assert tried > 200 and same / tried > 0.8


def test_decisions_are_distributions_with_legal_sizes():
    eng = Engine(seed=0)
    pre = [Act("btn", "raise", 3), Act("bb", "call")]
    spots = [("btn", "AhKd", "", [[]]),
             ("bb", "QsJs", "", [[Act("btn", "raise", 2.5)]]),
             ("btn", "AcQd", "Kh8d3c", [pre, [Act("bb", "check")]]),
             ("bb", "Ts9s", "Qh8s4s", [pre, [Act("bb", "check"), Act("btn", "bet", 4)]]),
             ("bb", "KdKs", "Kh8d3c2sJd", [pre, [Act("bb", "check"), Act("btn", "check")],
                                           [Act("bb", "check"), Act("btn", "check")], [Act("bb", "check"), Act("btn", "bet", 97)]])]
    for hero, hole, board, hist in spots:
        d = eng.decide(hero, hole, board, hist)
        assert abs(sum(d.probs.values()) - 1) < 1e-9
        assert d.to <= 100 + 1e-9
    assert eng.decide(*spots[2]).source == "solver"
    # Facing an all-in: no raise option left.
    d = eng.decide(*spots[4])
    assert set(d.probs) == {"fold", "call"} and d.action in ("fold", "call")


def test_matches_run_and_stay_within_stacks():
    eng = EngineAgent(Engine(seed=3))
    for opp in (Station(), Maniac(), EngineAgent(Engine(seed=4))):
        r = match(eng, opp, 50, seed=1)
        assert np.all(np.abs(r) <= 200 + 1e-9)


def test_fold_preflop_results():
    class Folder:
        def act(self, *a):
            return "fold", 0
    h = parse_cards("AhKd"), parse_cards("2c7d")
    board = parse_cards("3s4s5s6h9c")
    assert play_hand({"btn": Folder(), "bb": Station()}, {"btn": h[0], "bb": h[1]}, board) == -0.5
    assert play_hand({"btn": Station(), "bb": Folder()}, {"btn": h[0], "bb": h[1]}, board) == 1.0       # BB folds to the limp
