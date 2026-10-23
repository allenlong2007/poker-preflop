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
    d = Engine(pot_odds=True, seed=1).decide("btn", "AsKd", board, history, stack=stack)
    assert len(calls) == 1
    assert d.diagnostics[metric]["equity"] == 0.6
    assert d.diagnostics[metric]["range_combos"] > 0
    assert d.diagnostics["context"]["effective_stack_bb"] == stack
    assert d.diagnostics["adaptation"]["preflop_weight"] == 0
    assert d.diagnostics["adjustments"]
    assert d.label in d.diagnostics["legal_options"]
    json.dumps(d.diagnostics, allow_nan=False)


def test_preflop_reraise_logs_range_price_and_realization():
    eng = Engine(seed=0)
    eng.strict = 0.37
    history = [[Act("btn", "raise", 3), Act("bb", "raise", 9), Act("btn", "raise", 15)]]
    d = eng.decide("bb", "Ts9h", "", history, sample=False)
    evidence = d.diagnostics["reraise_war"]
    assert evidence["range_top_share"] == 0.06
    assert evidence["realization"] == 0.55
    assert evidence["price"] == pytest.approx(6 / 30)
    assert any(x["stage"] == "reraise_war" for x in d.diagnostics["adjustments"])
