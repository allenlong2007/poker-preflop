"""Checks for the PokerNow log parser and server (experiment 31)."""
import json
from pathlib import Path

import pytest

from bot.engine import POLICY
from bot.pokernow.parse import cards, describe, parse_hand, read_log

DEMO = Path(__file__).resolve().parents[1] / "bot" / "pokernow" / "demo_log.csv"
MSGS = read_log(DEMO.read_text())


def upto(text):
    """The demo log up to and including the first line containing `text`."""
    k = next(i for i, m in enumerate(MSGS) if text in m)
    return MSGS[:k + 1]


def test_cards_and_csv_order():
    assert cards("[A♠, 10♥, 2♦]") == ["As", "Th", "2d"]
    assert cards("K♥, 8♦, 3♣ [2♠]") == ["Kh", "8d", "3c", "2s"]
    assert MSGS[0].startswith("-- starting hand #1") and MSGS[-1] == "-- ending hand #2 --"


def test_latest_hand_seats_button_and_blinds():
    h = parse_hand(upto("Your hand is K♠"), hero_seat=3)
    t = describe(h)
    assert t["hand"] == 2 and t["heads_up"]
    assert t["button"] == "Friend" and not t["hero_has_button"]
    assert t["hero"]["seat"] == 3 and t["hero"]["position"] == "big blind"
    assert t["villain"]["seat"] == 7 and t["villain"]["position"] == "BTN / small blind"
    assert t["hero"]["stack_start"] == 19.90 and h.bb == 0.20 and h.sb == 0.10
    assert t["hero"]["bet_this_street"] == 0.20 and t["villain"]["bet_this_street"] == 0.10
    assert t["pot_main"] == 0 and t["pot_total"] == 0.30          # blinds are bets, not yet the pot
    assert t["to_act"] == "villain"                                 # the BTN acts first preflop


def test_bets_pot_and_to_call_are_separate():
    t = describe(parse_hand(upto("bets 1.80"), hero_seat=3))
    assert t["street"] == "turn" and t["board"] == ["Kh", "8d", "3c", "2s"]
    assert t["villain"]["bet_this_street"] == 1.80 and t["hero"]["bet_this_street"] == 0
    assert t["pot_main"] == 2.00                                    # 0.60 x 2 preflop + 0.40 x 2 flop
    assert t["pot_total"] == 3.80
    assert t["to_call"] == 1.80 and t["to_call_bb"] == 9.0
    assert t["hero"]["stack_now"] == 18.90 and t["to_act"] == "hero"


def test_history_in_big_blinds():
    h = parse_hand(upto("bets 4.10"), hero_seat=3)
    pre, flop, turn, river = h.history
    assert [(a.player, a.kind, a.to) for a in pre] == [("btn", "raise", 3.0), ("bb", "call", 0.0)]
    assert [a.kind for a in flop] == ["check", "bet", "call"] and flop[1].to == 2.0
    assert turn[1].to == 9.0 and abs(river[1].to - 20.5) < 1e-9


def test_seat_found_by_name_and_hand_over():
    h = parse_hand(MSGS, hero_name="Allen")
    assert h.hero_seat == 3 and h.over and describe(h)["to_act"] is None


def test_waits_for_the_next_card():
    h = parse_hand(upto("calls 1.80"), hero_seat=3)                # turn called, river not dealt yet
    assert h.to_act() is None


@pytest.mark.skipif(not (POLICY / "preflop.json").exists(), reason="match policies not built")
def test_server_decisions_through_the_demo_hand(tmp_path, monkeypatch):
    from bot.pokernow import server
    from bot.pokernow.server import decide
    monkeypatch.setattr(server, "LOG_DIR", tmp_path)            # keep the real session log clean
    seen = []
    for k in range(1, len(MSGS) + 1):
        text = "entry,at,order\n" + "".join(f'"{m.replace(chr(34), chr(34) * 2)}",t,{i}\n' for i, m in enumerate(MSGS[:k]))
        out = decide({"log": text, "seat": 3})
        json.dumps(out)                                             # the response must be JSON-serialisable
        if out["decision"]:
            seen.append((out["table"]["street"], out["decision"]["node"]))
    streets = [s for s, _ in seen]
    assert streets[0] == "preflop" and {"flop", "turn", "river"} <= set(streets)
    assert any(n.startswith("bb_vs_raise") for _, n in seen)


def test_live_json_log_format():
    """The format PokerNow's /games/<id>/log returned on a live test table (experiment 31)."""
    rows = [("-- starting hand #3 (id: lnmqlq2oeb8v)  No Limit Texas Hold'em (dealer: \"BotA @ dTOIcSeZN7\") --", 10),
            ('Player stacks: #1 "BotA @ dTOIcSeZN7" (20.00) | #2 "BotB @ KGEM_vB9Xd" (20.00)', 11),
            ("Your hand is J♣, 9♠", 12),
            ('"BotA @ dTOIcSeZN7" posts a small blind of 0.10', 13),
            ('"BotB @ KGEM_vB9Xd" posts a big blind of 0.20', 14),
            ('"BotA @ dTOIcSeZN7" calls 0.20', 15),
            ('"BotB @ KGEM_vB9Xd" raises to 0.80', 16)]
    # newest first, created_at as digit strings of different lengths
    text = json.dumps({"logs": [{"msg": m, "at": "x", "created_at": str(179212114741600 + c * 10 ** k)}
                                for k, (m, c) in reversed(list(enumerate(rows)))]})
    msgs = read_log(text)
    assert msgs[0].startswith("-- starting hand #3") and msgs[-1].endswith("raises to 0.80")
    t = describe(parse_hand(msgs, hero_seat=1))
    assert t["hero"]["position"] == "BTN / small blind" and t["to_act"] == "hero"
    assert t["to_call"] == 0.60 and t["pot_total"] == 1.00


def test_legal_bet_sizes():
    pre = ['-- starting hand #5 (id: x)  No Limit Texas Hold\'em (dealer: "A @ a1") --',
           'Player stacks: #1 "A @ a1" (20.00) | #2 "B @ b2" (3.00)',
           "Your hand is A♠, K♠", '"A @ a1" posts a small blind of 0.10', '"B @ b2" posts a big blind of 0.20']
    h = parse_hand(pre, hero_seat=1)
    assert h.legal_to(h.hero()) == (0.40, 20.00)                    # min-raise over the big blind
    h = parse_hand(pre + ['"A @ a1" calls 0.20', '"B @ b2" raises to 0.80'], hero_seat=1)
    assert h.legal_to(h.hero()) == (1.40, 20.00)                    # 0.80 + the 0.60 raise
    h = parse_hand(pre + ['"A @ a1" raises to 0.60', '"B @ b2" calls 0.60', "Flop:  [2♦, 7♣, 9♥]"], hero_seat=2)
    assert h.legal_to(h.hero()) == (0.20, 2.40)                     # first bet: at least 1bb; B has 2.40 left


@pytest.mark.skipif(not (POLICY / "preflop.json").exists(), reason="match policies not built")
def test_server_sizes_are_legal_and_all_in_flagged(tmp_path, monkeypatch):
    from bot.pokernow import server
    monkeypatch.setattr(server, "LOG_DIR", tmp_path)
    server.CACHE.clear()
    # B (big blind, 3.00 = 15bb) holds aces facing a 3bb open: the bot's 3-bet can't exceed 3.00.
    log = ['-- starting hand #9 (id: y)  No Limit Texas Hold\'em (dealer: "A @ a1") --',
           'Player stacks: #1 "A @ a1" (20.00) | #2 "B @ b2" (3.00)',
           "Your hand is A♠, A♥", '"A @ a1" posts a small blind of 0.10', '"B @ b2" posts a big blind of 0.20',
           '"A @ a1" raises to 0.60']
    for _ in range(20):                                            # sampling: check every outcome is legal
        server.CACHE.clear()
        d = server.decide({"log": json.dumps(log), "seat": 2})["decision"]
        if d["action"] in ("bet", "raise"):
            assert 1.00 <= d["to_chips"] <= 3.00
            assert d["all_in"] == (d["to_chips"] == 3.00)


def test_server_only_answers_the_pokernow_tab(tmp_path, monkeypatch):
    """Other sites open in the browser can reach 127.0.0.1 too: they must get nothing."""
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from bot.pokernow import server
    monkeypatch.setattr(server, "LOG_DIR", tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    server.PORT["n"] = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, origin, token):
        hdr = {"Content-Type": "application/json", "Origin": origin}
        if token:
            hdr["X-Pnbot-Token"] = token
        req = urllib.request.Request(url + path, data=json.dumps({"log": "[]", "seat": 1}).encode(), headers=hdr)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers.get("Access-Control-Allow-Origin")
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Access-Control-Allow-Origin")
    try:
        assert post("/decide", "https://www.pokernow.com", server.TOKEN) == (200, "https://www.pokernow.com")
        assert post("/decide", "https://www.pokernow.com", None)[0] == 403          # no token
        assert post("/decide", "https://evil.example", server.TOKEN) == (403, None)  # wrong site
        assert post("/acted", "https://evil.example", server.TOKEN)[0] == 403
        with urllib.request.urlopen(url + "/client.js") as r:                       # token goes into the script
            assert server.TOKEN in r.read().decode() and r.headers.get("Access-Control-Allow-Origin") is None
    finally:
        srv.shutdown()


def test_action_reports_are_validated():
    from bot.pokernow import server
    with pytest.raises(ValueError):
        server.acted({"ms": "nan"})
    with pytest.raises(ValueError):
        server.acted({"ms": 1e12})
