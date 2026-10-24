"""Opponent evidence must precede an exploit; repeated actions are clustered by hand."""
from bot.reads import evidence_profile, gate, interval, serving_profile


def hand(i, kind="raise"):
    return {"id": str(i), "opp_id": "villain", "hero_id": "hero", "opp_pos": "btn",
            "hero_pos": "bb", "stacks": {"hero": 100, "opp": 100}, "shown": {}, "board": "",
            "history": [[["btn", kind, 3 if kind == "raise" else None]]], "number": i}


def test_uncertainty_shrinks_and_sparse_extreme_read_abstains():
    assert interval(0, 0) == (0.0, 1.0)
    assert interval(30, 30)[1] - interval(30, 30)[0] > interval(3000, 3000)[1] - interval(3000, 3000)[0]
    p = evidence_profile([hand(i) for i in range(30)])
    assert not gate(p, "btn_first_in", {"raise": .5, "fold": .5})[1]["applied"]


def test_supported_read_is_conservative_and_duplicates_do_not_add_evidence():
    hands = [hand(i, "raise" if i % 10 else "fold") for i in range(3000)]
    p = evidence_profile(hands + hands)
    rates, report = gate(p, "btn_first_in", {"raise": .5, "fold": .5})
    assert report["n_hands"] == 3000 and report["applied"]
    assert .5 < rates["raise"] < .9
    assert abs(sum(rates.values()) - 1) < 1e-9
    assert not serving_profile({"hands": 9999, "nodes": p["nodes"]})["nodes"]


def test_duplicate_actions_and_selection_biased_showdowns_cannot_bypass_gate():
    h = hand(1)
    h["history"] += [[["bb", "bet", 1], ["btn", "raise", 3], ["bb", "raise", 6], ["btn", "raise", 12]]]
    p = evidence_profile([h] * 1000)
    assert p["read_evidence"]["nodes"]["flop_vs_bet"]["n"] == 1
    from bot.adapt import evidence_strictness, size_value
    assert evidence_strictness([] , prior=.01) == 0
    assert size_value({"nodes": {"shown_big": {"n": 10000, "value": 1}}})[1]["big"] == 1


def test_unknown_opponent_does_not_inherit_population_read():
    from bot.adapt import Adapter
    class Engine:
        def set_adaptation(self, *args):
            pass
    a = Adapter(Engine())
    a.pop = .01
    a.apply("new")
    assert a.engine.strict == 0
