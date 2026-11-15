"""Chipzen adapter: state translation, legal actions, and a full freezeout against a mini Chipzen-style dealer."""
import random

import pytest

chipzen = pytest.importorskip("chipzen")
from chipzen import Card, GameState  # noqa: E402

from bot.chipzen.adapter import Decider, big_blind, to_action, translate  # noqa: E402
from bot.chipzen.sim import deal_hand  # noqa: E402
from bot.engine import Decision  # noqa: E402


def gs(**kw):
    base = dict(hand_number=1, phase="preflop", hole_cards=[Card.from_str("Ah"), Card.from_str("Kd")], board=[],
                pot=30, your_stack=990, opponent_stacks=[980], your_seat=0, dealer_seat=0, to_call=10,
                min_raise=40, max_raise=1000, valid_actions=["fold", "call", "raise"], action_history=[])
    base.update(kw)
    return GameState(**base)


BLINDS = [{"seat": 0, "action": "post_small_blind", "amount": 10, "phase": "preflop"},
          {"seat": 1, "action": "post_big_blind", "amount": 20, "phase": "preflop"}]


def test_big_blind_reads_the_hand_posts_and_short_posts():
    assert big_blind(BLINDS) == 20
    short = [dict(BLINDS[0]), dict(BLINDS[1], amount=7)]
    assert big_blind(short) == 20                         # short all-in BB: the level is still 2 x SB
    assert big_blind([], fallback=50) == 50


def test_translate_button_open_and_effective_stack():
    s = gs(your_seat=1, dealer_seat=0, your_stack=980, opponent_stacks=[950], to_call=40,
           action_history=BLINDS + [{"seat": 0, "action": "raise", "amount": 50, "phase": "preflop"}])
    hero, hist, eff = translate(s, 20)
    assert hero == "bb"
    assert [(a.player, a.kind, a.to) for a in hist[0]] == [("btn", "raise", 2.5)]
    assert eff == pytest.approx(min(980 + 20, 950 + 50) / 20)


def test_translate_postflop_bet_then_raise_labels():
    h = BLINDS + [{"seat": 0, "action": "call", "amount": 10, "phase": "preflop"},
                  {"seat": 1, "action": "check", "amount": 0, "phase": "preflop"},
                  {"seat": 1, "action": "raise", "amount": 20, "phase": "flop"},
                  {"seat": 0, "action": "raise", "amount": 60, "phase": "flop"}]
    s = gs(phase="flop", board=[Card.from_str(c) for c in ("2c", "7d", "Qs")], your_seat=1, dealer_seat=0,
           your_stack=960, opponent_stacks=[920], action_history=h)
    hero, hist, eff = translate(s, 20)
    assert [(a.player, a.kind, a.to) for a in hist[1]] == [("bb", "bet", 1.0), ("btn", "raise", 3.0)]
    assert eff == pytest.approx(50.0)


def test_to_action_clamps_and_never_folds_a_free_check():
    s = gs(min_raise=40, max_raise=300, valid_actions=["fold", "call", "raise"])
    assert to_action(Decision("raise", 100.0, {}, "", ""), s, 20) == ("raise", 300)
    assert to_action(Decision("raise", 1.0, {}, "", ""), s, 20) == ("raise", 40)
    free = gs(to_call=0, valid_actions=["check", "raise"])
    assert to_action(Decision("fold", 0, {}, "", ""), free, 20) == ("check", 0)
    short = gs(valid_actions=["fold", "call"])
    assert to_action(Decision("raise", 5.0, {}, "", ""), short, 20) == ("call", 0)


def test_full_freezeout_with_escalating_blinds_stays_legal(tmp_path):
    rng = random.Random(7)
    decider = Decider(log_dir=tmp_path)
    stacks, sb, button = {0: 1000, 1: 1000}, 5, 0
    for hand in range(1, 300):
        if hand % 20 == 0:
            sb = int(sb * 1.5)
        stacks = deal_hand(rng, stacks, sb, button, {0: decider, 1: decider}, hand)
        assert sum(stacks.values()) == 2000
        if 0 in stacks.values():
            break
        button = 1 - button
    rows = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert rows and not any('"error"' in r for r in rows)


def test_runner_refuses_a_shared_or_linked_token(tmp_path):
    import os
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from chipzen_remote import read_token
    tok = tmp_path / "token"
    tok.write_text("cz_extbot_x\n")
    os.chmod(tok, 0o600)
    assert read_token(tok) == "cz_extbot_x"
    os.chmod(tok, 0o644)
    with pytest.raises(SystemExit):
        read_token(tok)
    link = tmp_path / "link"
    link.symlink_to(tok)
    with pytest.raises(OSError):
        read_token(link)
    assert read_token(tmp_path / "missing") == ""


def test_hand_record_feeds_the_opponent_profile():
    from bot.chipzen.adapter import hand_record
    from bot.opponent import profile
    result = {"hand_number": 3, "payouts": [{"seat": 1, "amount": 1000}],
              "showdown": [{"seat": 0, "hole_cards": ["Ah", "Kd"]}, {"seat": 1, "hole_cards": ["Qs", "Qc"]}],
              "action_history": BLINDS + [
                  {"seat": 0, "action": "raise", "amount": 50, "phase": "preflop"},
                  {"seat": 1, "action": "raise", "amount": 500, "phase": "preflop"},
                  {"seat": 0, "action": "call", "amount": 450, "phase": "preflop"}],
              "stacks": [500, 1500]}
    r = hand_record(result, 0, "AhKd", "2c7d9sJhTd", "PluriBot", "m1")
    assert r["hero_pos"] == "btn" and r["opp_pos"] == "bb" and r["bb"] == 20
    assert r["stacks"] == {"hero": 50.0, "opp": 50.0}
    assert r["net"]["hero"] == -25.0 and r["shown"] == {"hero": "AhKd", "opp": "QsQc"}
    assert r["history"][0] == [["btn", "raise", 2.5], ["bb", "raise", 25.0], ["btn", "call", None]] or \
        r["history"][0][1] == ["bb", "raise", 25.0]
    prof = profile([r])
    assert prof["nodes"]["bb_vs_open"]["n"] == 1


def test_micro_pushfold_calls_wide_and_shoves_short():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    e = v8_engine()
    d = e.decide("bb", "7h2h", "", [[Act("btn", "raise", 2.7)]], sample=False, stack=2.7)
    assert d.action == "call"
    d = e.decide("btn", "Kd7c", "", [[]], sample=False, stack=4.0)
    assert d.action == "raise" and d.to >= 3.99



def test_wide_3bet_is_opt_in_and_reads_min_3bets_as_wide():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    hist = [[Act("btn", "raise", 2.5), Act("bb", "raise", 4.0)]]           # a min-3-bet, no reads yet
    assert not v8_engine().wide_3bet
    e = v8_engine(wide_3bet=True)
    d = e.decide("btn", "KdJc", "", hist, sample=False, stack=100.0)
    assert d.action == "raise" and d.to > 4.0 and "wide_3bet" in d.diagnostics
    d = e.decide("btn", "7h2c", "", hist, sample=False, stack=100.0)
    assert not (d.action == "raise" and d.to > 4.0)                     # trash is not 4-bet
    normal = [[Act("btn", "raise", 2.5), Act("bb", "raise", 7.5)]]      # a 3x 3-bet starts from a tight prior
    assert "wide_3bet" not in e.decide("btn", "KdJc", "", normal, sample=False, stack=100.0).diagnostics
    e.opp_3bet = (16, 20)                                               # 3-bets 80% of our opens
    assert "wide_3bet" in e.decide("btn", "KdJc", "", normal, sample=False, stack=100.0).diagnostics


def test_decider_counts_the_opponents_3bets_per_opponent():
    from bot.chipzen.adapter import Decider, threebet_seen
    three = BLINDS + [{"seat": 0, "action": "raise", "amount": 50, "phase": "preflop"},
                      {"seat": 1, "action": "raise", "amount": 150, "phase": "preflop"}]
    fold = BLINDS + [{"seat": 0, "action": "raise", "amount": 50, "phase": "preflop"},
                     {"seat": 1, "action": "fold", "amount": 0, "phase": "preflop"}]
    assert threebet_seen(three, 0) == (1, 1) and threebet_seen(fold, 0) == (0, 1)
    assert threebet_seen(three, 1) == (0, 0)                            # only our button opens count
    d = Decider()
    d.set_opponent("PluriBot")
    d.observe({"action_history": three}, 0)
    d.observe({"action_history": fold}, 0)
    d.set_opponent("pebble")
    d.observe({"action_history": fold}, 0)
    assert d.reads == {"PluriBot": [1, 2], "pebble": [0, 1]}


def test_3bet_reads_persist_across_matches_from_hand_records(tmp_path):
    import json
    from bot.chipzen.adapter import Decider
    recs = [{"opp_name": "PluriBot", "hero_pos": "btn", "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5]]]},
            {"opp_name": "PluriBot", "hero_pos": "btn", "history": [[["btn", "raise", 2.5], ["bb", "fold", None]]]},
            {"opp_name": "PluriBot", "hero_pos": "bb", "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5]]]},
            {"opp_name": "pebble", "hero_pos": "btn", "history": [[["btn", "call", 1.0], ["bb", "raise", 4.0]]]}]
    (tmp_path / "hands.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")
    assert Decider(log_dir=tmp_path).reads == {"PluriBot": [1, 2]}
    assert Decider(log_dir=tmp_path / "empty").reads == {}
