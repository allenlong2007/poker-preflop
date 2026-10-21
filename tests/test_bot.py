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
    pre = [Act("btn", "raise", 2.25), Act("bb", "call")]
    spots = [("btn", "AhKd", "", [[]]),
             ("bb", "QsJs", "", [[Act("btn", "raise", 2.5)]]),
             ("btn", "AcQd", "Kh8d3c", [pre, [Act("bb", "check")]]),
             ("bb", "Ts9s", "Qh8s4s", [pre, [Act("bb", "check"), Act("btn", "bet", 4)]]),
             ("bb", "KdKs", "Kh8d3c2sJd", [pre, [Act("bb", "check"), Act("btn", "check")],
                                           [Act("bb", "check"), Act("btn", "check")], [Act("bb", "check"), Act("btn", "bet", 97.75)]])]
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


def test_size_weights_pseudo_harmonic():
    from bot.engine import size_weights
    sizes = (1 / 3, 1 / 2, 1.0)
    assert size_weights(0.5, sizes) == [(1, 1.0)]                 # a trained size maps to itself
    assert size_weights(0.1, sizes) == [(0, 1.0)]                 # below the range: the smallest size
    assert size_weights(3.0, sizes) == [(2, 1.0)]                 # above: the largest
    (a, wa), (b, wb) = size_weights(0.75, sizes)
    assert (a, b) == (1, 2) and abs(wa + wb - 1) < 1e-12
    assert abs(wa - (1.0 - 0.75) * 1.5 / (0.5 * 1.75)) < 1e-12    # (B - x)(1 + A) / ((B - A)(1 + x))
    assert size_weights(0.75, sizes, translate=False) == [(2, 1.0)]   # nearest on a log scale


def test_blended_decision_lies_between_neighbours():
    eng = Engine(seed=0)
    pre = [Act("btn", "raise", 2.25), Act("bb", "call")]
    flop = [Act("bb", "check"), Act("btn", "check")]
    spot = lambda b: eng.decide("bb", "Qc9d", "Kh8d3c2s", [pre, flop, [Act("bb", "check"), Act("btn", "bet", b)]],
                                sample=False).probs["fold"]
    lo, hi, mid = spot(3.0), spot(6.0), spot(4.5)                  # 1/2 pot, pot, 3/4 pot (pot 6bb)
    assert min(lo, hi) - 1e-9 <= mid <= max(lo, hi) + 1e-9


def test_no_deep_preflop_jam_over_a_3bet():
    """Experiment 36: at 100bb the BTN 4-bets small instead of jamming; all-ins only at short stacks."""
    eng = Engine(seed=0)
    pre = lambda t: [[Act("btn", "raise", 2.25), Act("bb", "raise", t)]]
    for hole in ("Ah4d", "KhKd", "AdKs", "As5s", "QsJs"):
        for t, stack in ((3.5, 79), (6.75, 100)):
            d = eng.decide("btn", hole, "", pre(t), sample=False, stack=stack)
            assert d.probs.get("all-in", 0) < 1e-6, (hole, t, d.probs)
    d = eng.decide("btn", "KhKd", "", pre(6.75), sample=False)
    assert d.label == "4-bet" and abs(d.to - 2.2 * 6.75) < 0.01
    d = eng.decide("btn", "KhKd", "", pre(6.75), sample=False, stack=25)
    assert d.probs.get("all-in", 0) > 0.5
    # The BB answers a small 4-bet with fold / call / all-in; the BTN answers the 5-bet jam with fold / call.
    four = [[Act("btn", "raise", 2.25), Act("bb", "raise", 6.75), Act("btn", "raise", 14.85)]]
    assert set(eng.decide("bb", "AhAd", "", four).probs) <= {"fold", "call", "all-in"}
    five = [four[0] + [Act("bb", "raise", 100)]]
    assert set(eng.decide("btn", "QhQd", "", five).probs) <= {"fold", "call"}


def test_adaptation_never_adds_deep_all_ins():
    """Experiment 37: a best response to an extreme opponent model stays off 100bb shoves."""
    from bot.adapt import Preflop
    pre = Preflop()
    prof = {"hands": 300, "nodes": {
        "btn_first_in": {"n": 150, "limp": 0.6, "raise": 0.3, "fold": 0.1},
        "bb_vs_limp": {"n": 90, "raise": 0.9, "check": 0.1},
        "bb_vs_open": {"n": 60, "fold": 0.7, "call": 0.2, "raise": 0.1}}}
    out = pre.adapt(prof)
    assert out and all(0 < u["confidence"] <= 0.85 for u in out["locked"].values())
    for name, acts in out["nodes"].items():
        # The bot's answers come from a tree without its all-ins (except where fold / call is all that's left).
        from preflop.shortstack import JAM_ONLY_RAISE
        assert "all-in" not in acts or len(acts) == 2 or name in JAM_ONLY_RAISE, name


def test_river_calls_depend_on_the_raise_size():
    """Experiments 40-41: one pair folds more to a bigger river raise; top two pair always calls (against an
    opponent read as value-heavy, strictness 1)."""
    eng = Engine(seed=0, pot_odds=True)
    eng.strict = 1.0
    line = lambda r: [[Act("btn", "call"), Act("bb", "raise", 5), Act("btn", "call")],
                      [Act("bb", "check"), Act("btn", "check")], [Act("bb", "bet", 7.5), Act("btn", "call")],
                      [Act("bb", "bet", 12.5), Act("btn", "raise", r)]]
    board = "8h2hAhJs9s"
    small = eng.decide("bb", "TsJh", board, line(25), sample=False).probs["call"]
    big = eng.decide("bb", "TsJh", board, line(62.5), sample=False).probs["call"]
    assert small > big + 0.1
    assert eng.decide("bb", "AcJd", board, line(62.5), sample=False).probs["call"] > 0.85   # flushes and straights possible


def test_short_stack_postflop_jams_or_checks():
    """Experiment 40: with ~5bb behind, no small c-bets: shove with equity, check without it."""
    eng = Engine(seed=0)
    hist = [[Act("btn", "raise", 2.25), Act("bb", "call")], [Act("bb", "check")]]
    strong = eng.decide("btn", "KsKd", "Kc7d2s", hist, sample=False, stack=7)
    weak = eng.decide("btn", "4h3d", "KcQdJs", hist, sample=False, stack=7)
    assert strong.label == "all-in" and strong.source == "short"
    assert weak.label == "check"


def test_every_depth_can_answer_a_shove():
    """The 40 / 60bb trees have no shoves of their own; facing one they must still answer (100bb nodes)."""
    eng = Engine(seed=0)
    for st in (100, 60, 45, 40, 30, 25, 15, 10, 6):
        for hero, hist in (("bb", [[Act("btn", "raise", st)]]),
                           ("btn", [[Act("btn", "raise", 2.25), Act("bb", "raise", st)]]),
                           ("bb", [[Act("btn", "call"), Act("bb", "raise", 5), Act("btn", "raise", st)]]),
                           ("bb", [[Act("btn", "raise", 2.25), Act("bb", "raise", 6.75), Act("btn", "raise", st)]])):
            d = eng.decide(hero, "QhQd", "", hist, sample=False, stack=st)
            assert set(d.probs) <= {"fold", "call"} and abs(sum(d.probs.values()) - 1) < 1e-9


def test_never_just_call_with_the_nuts():
    """Experiments 43-44: straight flush, quads, the nut flush: bet or raise, never call or check."""
    eng = Engine(seed=0, pot_odds=True)
    pre = [Act("btn", "raise", 2.25), Act("bb", "call")]
    riv = [pre, [Act("bb", "check"), Act("btn", "check")], [Act("bb", "check"), Act("btn", "check")], [Act("bb", "bet", 3)]]
    d = eng.decide("btn", "9h8h", "7h6h5hKc2d", riv, sample=False)
    assert d.label == "raise all-in"
    for seat, hole, board, hist in (("btn", "KdKs", "KhKc5h", [pre, [Act("bb", "check")]]),
                                    ("bb", "KdKs", "KhKc5h", [pre, [Act("bb", "check"), Act("btn", "bet", 1.5)]]),
                                    ("bb", "AhJh", "Kh8h3h2c", [pre, [Act("bb", "check"), Act("btn", "bet", 1.5), Act("bb", "call")],
                                                                [Act("bb", "check"), Act("btn", "bet", 5)]])):
        assert eng.decide(seat, hole, board, hist, sample=False).action in ("bet", "raise")
