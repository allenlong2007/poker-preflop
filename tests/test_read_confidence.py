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


def test_temporal_calibration_keeps_selected_reveals_in_shadow(monkeypatch):
    import numpy as np
    from bot import bluffmodel
    monkeypatch.setattr(bluffmodel, "prior", lambda: bluffmodel.Model(np.zeros(10)))
    seen = []
    def examples(r, who):
        seen.append(r["id"])
        return [(np.ones(10), float(int(r["id"]) % 2))] * 3
    monkeypatch.setattr(bluffmodel, "examples_from_record", examples)
    hands = [hand(i) for i in range(100)]
    report = bluffmodel.calibration_report(hands + hands)
    assert report["train_hands"] == 50 and report["validation_hands"] == 50
    assert report["applied"] is False and report["reason"] == "selected_reveals_not_range_calibrated"
    assert len(seen) == 100
    assert report["brier_improvement_interval"][0] <= 0 <= report["brier_improvement_interval"][1]
    model, n = bluffmodel.for_opponent(hands)
    assert np.array_equal(model.w, np.zeros(10))


def test_mixed_opponents_rejected():
    import pytest
    other = dict(hand(2), opp_id="different")
    with pytest.raises(ValueError, match="identities"):
        evidence_profile([hand(1), other])


def test_other_preflop_actions_cannot_qualify_shove_read():
    from bot.engine import Act, Engine
    e = Engine(pot_odds=True)
    from bot.reads import jam_key
    e.read_evidence = {"version": 1, "decisions": {}, "jams": {
        jam_key("btn_first_in", 100, []): {"n": 3000, "hits": 90, "index": 1}}}
    d = e.decide("bb", "Ah9c", "", [[Act("btn", "raise", 100)]], sample=False)
    assert not d.shove_read


def test_postflop_baseline_uses_same_nodes_and_weights():
    from bot.adapt import postflop_exploit
    prof = {"nodes": {"flop_vs_bet": {"n": 3000, "fold": .6}}}
    base = {"nodes": {"flop_vs_bet": {"n": 100, "fold": .4},
                      "river_vs_bet": {"n": 10000, "fold": .9}}}
    assert postflop_exploit(prof, base)["bluff"] > 1


def test_near_jams_do_not_count_as_exact_jams_and_depths_are_separate():
    from bot.engine import Act, Engine
    hands = [hand(i) for i in range(3000)]
    for h in hands:
        h["history"][0][0][2] = 60
    p = evidence_profile(hands)
    e = Engine(pot_odds=True)
    e.read_evidence = p["read_evidence"]
    d = e.decide("bb", "Ah9c", "", [[Act("btn", "raise", 100)]], sample=False)
    assert not d.shove_read
    for h in hands:
        h["history"][0][0][2] = 100
    e.read_evidence = evidence_profile(hands)["read_evidence"]
    assert e.decide("bb", "Ah9c", "", [[Act("btn", "raise", 100)]], sample=False).shove_read
    assert not e.decide("bb", "Ah9c", "", [[Act("btn", "raise", 40)]], stack=40, sample=False).shove_read
