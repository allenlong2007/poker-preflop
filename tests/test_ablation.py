"""Experiment 50: stage switches, held-out opponents and simulated reads for ablation matches."""
import json

import numpy as np

from bot.engine import Act, Engine
from bot.heldout import HELDOUT, BaseMirror, make
from bot.selfplay import DECK, play_hand

STAGES = BaseMirror.STAGES


def test_switched_off_stages_never_change_a_decision():
    hist = [[Act("btn", "raise", 2.25), Act("bb", "call")], [Act("bb", "check"), Act("btn", "bet", 3.0)]]
    eng = Engine(exact_required=False, seed=0, pot_odds=True)
    eng.off = set(STAGES)
    d = eng.decide("bb", "AsAd", "Ah7c2d", hist, sample=False)
    assert not d.diagnostics.get("adjustments") or \
        {a["stage"] for a in d.diagnostics["adjustments"]} <= {"exploit_and_legality"}
    on = Engine(exact_required=False, seed=0, pot_odds=True).decide("bb", "AsAd", "Ah7c2d", hist, sample=False)
    assert {a["stage"] for a in on.diagnostics["adjustments"]} - {"exploit_and_legality"}   # some stage acts here


def test_held_out_bots_are_deterministic_and_legal():
    rng = np.random.default_rng(0)
    deals = [[DECK[i] for i in rng.choice(52, size=9, replace=False)] for _ in range(6)]
    for name in HELDOUT:
        runs = []
        for _ in range(2):
            a, b = make(name), make("geo")
            for stack in (100.0, 20.0):
                a.stack = b.stack = stack
                runs.append([play_hand({"btn": a, "bb": b}, {"btn": c[:2], "bb": c[2:4]}, c[4:], stack=stack)
                             for c in deals])
        assert runs[:2] == runs[2:]                                 # same seed, same deals: same results
        assert all(abs(x) <= 100 for r in runs for x in r)


def test_simulated_reads_pass_the_live_evidence_gates():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from ablation import read
    _, state = read(("limper", 200))
    # The limper never bets when checked to: with identities attached the gate approves that node.
    assert state["evidence"]["nodes"]
    assert "flop_bet_when_checked_to" in state["profile"]["nodes"]
    json.dumps(state["evidence"])                                  # the evidence report stays serialisable


def test_defend_scale_is_per_street():
    from bot.adapt import postflop_exploit
    base = {"nodes": {k: {"n": 500, "bet": 0.3, "check": 0.7}
                      for k in ("flop_cbet", "turn_bet_when_checked_to", "river_bet_when_checked_to")}}
    always_cbets = {"hands": 300, "nodes": {"flop_cbet": {"n": 300, "bet": 1.0, "check": 0.0},
                                            "turn_bet_when_checked_to": {"n": 300, "bet": 0.3, "check": 0.7}},
                    "read_evidence": {"version": 1, "decisions": {"flop_cbet": {"applied": True},
                                                                  "turn_bet_when_checked_to": {"applied": True}}}}
    post = postflop_exploit(always_cbets, base)
    assert post["defend"] > 1.5                                  # pooled: looks aggressive everywhere
    assert post["defend_street"]["flop"] >= 2.0 and abs(post["defend_street"]["turn"] - 1) < 0.01
    assert post["defend_street"]["river"] == 1.0                 # no evidence there: no adjustment
    eng = Engine(exact_required=False, seed=0)
    eng.set_adaptation({}, 0.0, post)
    assert eng._defend(1) >= 2.0 and eng._defend(2) < 1.01 and eng._defend(3) == 1.0


def test_results_do_not_depend_on_the_hash_seed():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; sys.path[:0]=['.','scripts']; import ablation as ab; "
            "print(ab.run(('full', 100.0, 'geo', 25))[1].sum())")
    out = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True,
                          env={"PYTHONHASHSEED": str(h), "PATH": "/usr/bin:/bin"}).stdout for h in (1, 2)}
    assert len(out) == 1


def test_pot_odds_check_only_at_shallow_stacks():
    from bot.engine import POT_ODDS_MAX_BB
    hist = [[Act("btn", "raise", 2.25), Act("bb", "call")], [Act("bb", "check"), Act("btn", "check")],
            [Act("bb", "check"), Act("btn", "check")], [Act("bb", "bet", 3)]]
    eng = Engine(exact_required=False, pot_odds=True, seed=1)
    deep = eng.decide("btn", "AsKd", "Kh8d3c2s9h", hist, stack=100)
    shallow = eng.decide("btn", "AsKd", "Kh8d3c2s9h", hist, stack=POT_ODDS_MAX_BB)
    assert "pot_odds" not in deep.diagnostics and "pot_odds" in shallow.diagnostics
