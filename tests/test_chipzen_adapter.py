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
    k = "pid:p_abc"
    recs = [{"opp_key": k, "hero_pos": "btn", "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5]]]},
            {"opp_key": k, "hero_pos": "btn", "history": [[["btn", "raise", 2.5], ["bb", "fold", None]]]},
            {"opp_key": k, "hero_pos": "bb", "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5]]]},
            {"opp_key": "pid:p_x", "hero_pos": "btn", "history": [[["btn", "call", 1.0], ["bb", "raise", 4.0]]]},
            {"opp_name": "PluriBot", "hero_pos": "btn", "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5]]]}]
    (tmp_path / "hands.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n[1]\n")
    assert Decider(log_dir=tmp_path).reads == {k: [1, 2], "name:PluriBot": [1, 1]}   # a name keys old records
    assert Decider(log_dir=tmp_path / "empty").reads == {}
    (tmp_path / "hands.jsonl").write_text("\n".join(json.dumps(recs[0]) for _ in range(90)) + "\n")
    assert Decider(log_dir=tmp_path).reads == {k: [30, 30]}              # carried-over counts are capped
    d = Decider()
    d.set_opponent(None)
    a = d.opp
    d.set_opponent(None)
    assert a != d.opp                                                    # no id: never shares a read


def test_reraise_defend_prices_calls_after_our_raise_at_any_depth():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    hist = [[Act("btn", "raise", 2.5), Act("bb", "call", None)],
            [Act("bb", "check", None), Act("btn", "bet", 1.7), Act("bb", "raise", 6.0)]]
    off = v8_engine().decide("btn", "QhJh", "Qd7c2s", hist, sample=False, stack=100.0)
    on = v8_engine(reraise_defend=True).decide("btn", "QhJh", "Qd7c2s", hist, sample=False, stack=100.0)
    assert "pot_odds" not in off.diagnostics and "pot_odds" in on.diagnostics
    plain = [[Act("btn", "raise", 2.5), Act("bb", "call", None)], [Act("bb", "bet", 3.0)]]  # no bet of ours first
    assert "pot_odds" not in v8_engine(reraise_defend=True).decide(
        "btn", "QhJh", "Qd7c2s", plain, sample=False, stack=100.0).diagnostics


def test_load_reads_skips_overlong_lines_without_losing_the_next(tmp_path, monkeypatch):
    import json
    from bot.chipzen import adapter
    monkeypatch.setattr(adapter, "READS_MAX_LINE", 200)
    good = json.dumps({"opp_key": "pid:p", "hero_pos": "btn", "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5]]]})
    (tmp_path / "hands.jsonl").write_text("x" * 5000 + "\n" + good + "\n")
    assert adapter.load_reads(tmp_path / "hands.jsonl") == {"pid:p": [1, 1]}
    long_name = adapter.hand_record({"hand_number": 1, "action_history": BLINDS, "stacks": [990, 1010]},
                                    0, "AhKd", "", "n" * 10_000, "m")
    assert long_name is None or len(long_name["opp_name"]) <= 64


def test_aggro_defend_is_opt_in_and_prices_calls_vs_a_frequent_bettor_at_depth():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    plain = [[Act("btn", "raise", 2.5), Act("bb", "call", None)], [Act("bb", "bet", 3.0)]]
    off = v8_engine().decide("btn", "QhJh", "Qd7c2s", plain, sample=False, stack=100.0)
    assert "pot_odds" not in off.diagnostics
    e = v8_engine(aggro_defend=True)
    assert "pot_odds" not in e.decide("btn", "QhJh", "Qd7c2s", plain, sample=False, stack=100.0).diagnostics
    e.opp_aggr = (5, 50)                                # a passive opponent: still off at depth
    assert "pot_odds" not in e.decide("btn", "QhJh", "Qd7c2s", plain, sample=False, stack=100.0).diagnostics
    e.opp_aggr = (45, 50)                               # bets or raises 90% of the time
    on = e.decide("btn", "QhJh", "Qd7c2s", plain, sample=False, stack=100.0)
    assert "pot_odds" in on.diagnostics and on.probs.get("fold", 0) <= off.probs.get("fold", 0) + 1e-9


def test_aggression_reads_count_postflop_bets_and_persist(tmp_path):
    import json
    from bot.chipzen.adapter import aggression_seen, load_reads
    hist = [{"seat": 0, "action": "post_small_blind", "amount": 50, "phase": "preflop"},
            {"seat": 1, "action": "raise", "amount": 300, "phase": "preflop"},
            {"seat": 1, "action": "raise", "amount": 400, "phase": "flop"},
            {"seat": 0, "action": "call", "amount": 400, "phase": "flop"},
            {"seat": 1, "action": "check", "amount": 0, "phase": "turn"},
            {"seat": 1, "action": "all_in", "amount": 900, "phase": "river"}, "junk"]
    assert aggression_seen(hist, 0) == (2, 3)
    rec = {"opp_key": "pid:p", "hero_pos": "btn",
           "history": [[["btn", "raise", 2.5], ["bb", "call", None]], [["bb", "bet", 3.0], ["btn", "call", None]],
                       [["bb", "check", None], ["btn", "check", None]]]}
    (tmp_path / "hands.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec | {"opp_key": None}) + "\n")
    aggr = {}
    load_reads(tmp_path / "hands.jsonl", aggr)
    assert aggr == {"pid:p": [1, 2]}


def test_aggression_read_counts_an_allin_call_as_passive_and_caps_in_session():
    from bot.chipzen.adapter import AGGR_SEED_CAP, Decider, aggression_seen
    hist = [{"seat": 0, "action": "raise", "amount": 400, "phase": "flop"},
            {"seat": 1, "action": "all_in", "amount": 300, "phase": "flop"}]
    assert aggression_seen(hist, 0) == (0, 0)
    assert aggression_seen([{"seat": 1, "action": "raise", "phase": ["flop"]},
                            {"seat": 1, "action": "raise", "phase": "flop"}], 0) == (1, 1)
    d = Decider()
    d.set_opponent("pid:x")
    bets = [{"seat": 1, "action": "raise", "amount": 100, "phase": "flop"}] * 10
    for _ in range(20):
        d.observe({"action_history": bets}, 0)
    assert d.aggr["pid:x"][1] == AGGR_SEED_CAP and abs(d.aggr["pid:x"][0] - AGGR_SEED_CAP) < 1e-9


def test_name_keys_skip_possibly_truncated_names(tmp_path):
    import json
    from bot.chipzen.adapter import load_reads
    rec = {"hero_pos": "btn", "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5]]]}
    (tmp_path / "hands.jsonl").write_text(json.dumps(rec | {"opp_name": "x" * 64}) + "\n"
                                          + json.dumps(rec | {"opp_name": "ok"}) + "\n")
    assert load_reads(tmp_path / "hands.jsonl") == {"name:ok": [1, 1]}


def test_deep_jam_is_opt_in_and_folds_weak_hands_to_a_deep_jam():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    hist = [[Act("btn", "raise", 2.5), Act("bb", "raise", 7.5), Act("btn", "raise", 104.0)]]
    assert v8_engine().decide("bb", "KcTd", "", hist, sample=False, stack=96.0).action == "call"   # the old leak
    e = v8_engine(deep_jam=True)
    assert e.decide("bb", "KcTd", "", hist, sample=False, stack=96.0).action == "fold"
    assert e.decide("bb", "QsQd", "", hist, sample=False, stack=96.0).action == "call"
    e.opp_jams = (8, 12)                                # a jam-happy opponent: call wider
    assert e.decide("bb", "KcTd", "", hist, sample=False, stack=96.0).action == "call"
    short = [[Act("btn", "raise", 2.0), Act("bb", "raise", 6.0), Act("btn", "raise", 20.0)]]
    assert "deep_jam" not in v8_engine(deep_jam=True).decide("bb", "KcTd", "", short, sample=False,
                                                            stack=20.0).diagnostics   # shallow: the trees decide


def test_jam_seen_counts_deep_preflop_jams():
    from bot.chipzen.adapter import jam_seen
    rec = {"hero_pos": "bb", "stacks": {"hero": 96.0, "opp": 104.0},
           "history": [[["btn", "raise", 2.5], ["bb", "raise", 7.5], ["btn", "raise", 104.0], ["bb", "fold", None]]]}
    assert jam_seen(rec) == (1, 2)
    assert jam_seen(rec | {"stacks": {"hero": 20.0, "opp": 104.0}}) == (0, 0)
    assert jam_seen({"junk": 1}) == (0, 0)


def test_deep_jam_prices_a_big_4bet_that_borrowed_a_shallow_node():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    hist = [[Act("btn", "raise", 2.25), Act("bb", "raise", 7.5), Act("btn", "raise", 30.0)]]   # 30 of 50: not all-in
    assert v8_engine().decide("bb", "KcTd", "", hist, sample=False, stack=50.0).action == "call"   # the old leak
    e = v8_engine(deep_jam=True)
    d = e.decide("bb", "KcTd", "", hist, sample=False, stack=50.0)
    assert d.action == "fold" and d.diagnostics["deep_jam"]["borrowed"]
    assert e.decide("bb", "KsKd", "", hist, sample=False, stack=50.0).action != "fold"
    e.deepjam_borrow_frac = 0.0                          # true all-ins only: the old behaviour here
    assert e.decide("bb", "KcTd", "", hist, sample=False, stack=50.0).action == "call"


def test_fold3_seen_counts_only_our_bb_3bet_vs_their_open():
    from bot.chipzen.adapter import fold3_seen
    pre = lambda *acts: [{"phase": "preflop", "seat": s, "action": a} for s, a in acts]
    base = [(1, "post_small_blind"), (0, "post_big_blind")]
    assert fold3_seen(pre(*base, (1, "raise"), (0, "raise"), (1, "fold")), 0) == (1, 1)
    assert fold3_seen(pre(*base, (1, "raise"), (0, "raise"), (1, "call")), 0) == (0, 1)
    assert fold3_seen(pre(*base, (1, "raise"), (0, "call")), 0) == (0, 0)
    assert fold3_seen(pre(*base, (1, "call"), (0, "raise"), (1, "fold")), 0) == (0, 0)
    assert fold3_seen(pre(*base, (1, "raise"), (0, "raise"), (1, "fold")), 1) == (0, 0)


def test_light_3bet_is_opt_in_and_needs_a_folding_read():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    hist = [[Act("btn", "raise", 2.5)], [], [], []]
    off = v8_engine()
    on = v8_engine(light_3bet=True)
    on.opp_f3 = (1, 2)                                   # thin read: shrunk share 0.5, below light3_on
    d = on.decide("bb", "7c2d", "", hist, sample=False)
    assert "light_3bet" not in " ".join(d.notes)
    on.opp_f3 = (19, 20)                                 # folds 95%: 72o becomes a 3-bet
    d = on.decide("bb", "7c2d", "", hist, sample=False)
    assert d.action == "raise" and any("folds to 3-bets" in n for n in d.notes)
    assert off.decide("bb", "7c2d", "", hist, sample=False).action == "fold"


def test_fold3_from_record_skips_jams_and_short_stacks():
    from bot.chipzen.adapter import fold3_from_record
    rec = lambda to, st=100.0, kind="fold": {"hero_pos": "bb", "stacks": {"hero": st, "opp": st},
                                             "history": [[["btn", "raise", 2.5], ["bb", "raise", to], ["btn", kind, 0]]]}
    assert fold3_from_record(rec(9.0)) == (1, 1)
    assert fold3_from_record(rec(9.0, kind="call")) == (0, 1)
    assert fold3_from_record(rec(100.0)) == (0, 0)          # our 3-bet was a shove
    assert fold3_from_record(rec(5.0, st=10.0)) == (0, 0)   # too short for light_3bet
    assert fold3_from_record({}) == (0, 0)


def test_river_air_is_opt_in_and_folds_high_card_when_our_river_bet_is_raised():
    from bot.chipzen.adapter import v8_engine
    hist = [[("btn", "call", 1.0), ("bb", "check", 1.0)], [("bb", "check", 0), ("btn", "check", 0)],
            [("bb", "bet", 2.0), ("btn", "call", 2.0)], [("bb", "bet", 9.0), ("btn", "raise", 21.0)]]
    kw = dict(wide_3bet=True, reraise_defend=True, aggro_defend=True, deep_jam=True, light_3bet=True)
    on = v8_engine(river_air=True, **kw)
    d = on.decide("bb", "8s6h", "2hQsJsTcKd", hist, sample=False, stack=24.36)
    assert d.label == "fold" and not d.probs.get("call")
    d = on.decide("bb", "Ks6h", "2hQsJsTcKd", hist, sample=False, stack=24.36)    # a pair of our own: untouched
    assert d.probs.get("call", 0) > 0
    off = v8_engine(**kw)
    assert off.decide("bb", "8s6h", "2hQsJsTcKd", hist, sample=False, stack=24.36).probs.get("call", 0) > 0
    # mode 1 leaves a plain river bet alone; mode 2 folds high card below a king to it
    bet = hist[:3] + [[("bb", "check", 0), ("btn", "bet", 9.0)]]
    assert "river air" not in " ".join(on.decide("bb", "8s6h", "2hQsJsTcKd", bet, sample=False, stack=24.36).notes)
    on.river_air_mode = 2
    assert not on.decide("bb", "8s6h", "2hQsJsTcKd", bet, sample=False, stack=24.36).probs.get("call")


def test_scout_seeds_add_a_valid_fold_read_and_skip_bad_entries(tmp_path):
    import json
    from bot.chipzen.adapter import Decider, v8_engine, SCOUT_FILE
    (tmp_path / SCOUT_FILE).write_text(json.dumps({
        "name:alizabethbot3s": {"f3": [12, 13]},
        "name:bad1": {"f3": [5, 3]},            # folds > faced
        "name:bad2": {"f3": [1.5, 3]},          # not whole numbers
        "name:bad3": {"f3": [40, 40]},          # over the cap
        "pid:x": {"f3": [3, 3]},                # not a name key
        "name:unknown": {"f3": [3, 3]},         # sentinel names are never seeded
    }))
    d = Decider(v8_engine(light_3bet=True), log_dir=tmp_path)
    assert d.f3 == {"name:alizabethbot3s": [12, 13]}
    d.engine.opp_f3 = tuple(d.f3["name:alizabethbot3s"])
    dec = d.engine.decide("bb", "7c2d", "", [[("btn", "raise", 2.5)], [], [], []], sample=False)
    assert dec.action == "raise"                # 12/13 shrinks to 0.79, above light3_on
    (tmp_path / SCOUT_FILE).write_text("{not json")
    assert Decider(v8_engine(), log_dir=tmp_path).f3 == {}
    (tmp_path / SCOUT_FILE).write_text(json.dumps({"name:x": {"3b": [8, 29], "aggr": [45, 60], "f3": [1, 1]},
                                                   "name:y": {"aggr": [70, 70], "3b": "bad"}}))
    d = Decider(v8_engine(), log_dir=tmp_path)
    assert d.reads == {"name:x": [8, 29]} and d.aggr == {"name:x": [45, 60]} and d.f3 == {"name:x": [1, 1]}
