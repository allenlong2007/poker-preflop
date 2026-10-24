"""Decision diagnostics are numeric evidence from the calculation already performed."""
import json

import pytest

from bot import potodds
from bot.engine import Act, Engine


@pytest.mark.parametrize("stack,board,history,metric", [
    (100, "Kh8d3c2s9h", [[Act("btn", "raise", 2.25), Act("bb", "call")],
                         [Act("bb", "check"), Act("btn", "check")],
                         [Act("bb", "check"), Act("btn", "check")],
                         [Act("bb", "bet", 3)]], "pot_odds"),
    (10, "Kh8d3c", [[Act("btn", "raise", 2.25), Act("bb", "call")],
                     [Act("bb", "check")]], "short_stack"),
])
def test_equity_is_logged_without_an_extra_calculation(monkeypatch, stack, board, history, metric):
    calls = []

    def equity(*args):
        calls.append(args)
        return 0.6

    monkeypatch.setattr(potodds, "equity", equity)
    d = Engine(exact_required=False, pot_odds=True, seed=1).decide("btn", "AsKd", board, history, stack=stack)
    assert len(calls) == 1
    assert d.diagnostics[metric]["equity"] == 0.6
    assert d.diagnostics[metric]["range_combos"] > 0
    assert d.diagnostics["context"]["effective_stack_bb"] == stack
    assert d.diagnostics["adaptation"]["preflop_weight"] == 0
    assert d.diagnostics["adjustments"]
    assert d.label in d.diagnostics["legal_options"]
    json.dumps(d.diagnostics, allow_nan=False)


def test_preflop_reraise_logs_range_price_and_realization():
    eng = Engine(exact_required=False, seed=0)
    eng.strict = 0.37
    history = [[Act("btn", "raise", 3), Act("bb", "raise", 9), Act("btn", "raise", 15)]]
    d = eng.decide("bb", "Ts9h", "", history, sample=False)
    evidence = d.diagnostics["reraise_war"]
    assert evidence["range_top_share"] == 0.06
    assert evidence["realization"] == 0.55
    assert evidence["price"] == pytest.approx(6 / 30)
    assert any(x["stage"] == "reraise_war" for x in d.diagnostics["adjustments"])


def test_logging_works_without_an_optional_bluff_prior(monkeypatch):
    from bot import bluffmodel

    monkeypatch.setattr(bluffmodel, "prior", lambda: None)
    eng = Engine(exact_required=False, pot_odds=True, seed=1)
    history = [[Act("btn", "raise", 2.25), Act("bb", "call")], [Act("bb", "bet", 1)]]
    d = eng.decide("btn", "AsKd", "Kh8d3c", history)
    assert d.diagnostics["adaptation"]["bluff_model"] is None


def test_preflop_adaptation_preserves_the_lookup_baseline():
    eng = Engine(exact_required=False, seed=1)
    eng.lam = 0.5
    eng.adapted = {"btn_root": {a: [float(a == "fold")] * 169 for a in eng.pre["nodes"]["btn_root"]}}
    d = eng.decide("btn", "AsAh", "", [[]], sample=False)
    trace = next(x for x in d.diagnostics["adjustments"] if x["stage"] == "preflop_adaptation")
    assert trace["before"] == d.diagnostics["base"]["probs"]
    assert trace["after"]["fold"] > trace["before"]["fold"]
