"""Local server between the PokerNow tab and the engine (experiment 31).

    python -m bot.pokernow.server [port]          (default 8765)

Then, in the browser tab with the PokerNow game, run the bookmarklet printed at startup (or paste
`client.js` into the developer console). The page script reads the game log, sends it here, and shows the
table and the bot's recommendation in an overlay. It is advice only: it never clicks; you play the moves.

Endpoints:
    GET  /client.js       the page script
    GET  /games/demo      a mock table that replays a sample hand step by step (to check the overlay)
    POST /decide          {"log": <CSV or JSON text>, "seat": N | null, "you_name": str | null}
                          -> {"table": <labeled snapshot>, "decision": {...} | null, "wait": reason | null}
Every decision is appended to output/pokernow/decisions.jsonl for review after the session.
"""
import collections
import hashlib
import json
import math
import os
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bot.adapt import Adapter
from bot.engine import Engine
from bot.opponent import load, profile, read_style, record, results, split_hands, store
from bot.pokernow.parse import describe, parse_hand, read_log

ROOT = Path(__file__).resolve().parents[2]
CLIENT = Path(__file__).with_name("client.js")
LOG_DIR = ROOT / "output" / "pokernow"
ENGINE = Engine(pot_odds=True)   # experiment 39: every call is checked against the price
ADAPT = Adapter(ENGINE)            # experiment 37: re-fits the bot to the opponent as hands finish
DECIDE_LOCK = threading.Lock()
SEEN_HANDS = set()                 # hand ids already stored this run
CACHE = {}                         # (hand, hash of its log, actions so far) -> decision, so polling doesn't resample
# Only the PokerNow tab may use the server: every request must come from an allowed origin and carry this
# per-run token (it is written into client.js when the page loads it). Other sites open in the same browser
# can reach 127.0.0.1 too, and must not be able to feed logs in or read decisions out.
def _load_token():
    """A secret kept in a private file, so the bookmarklet (which carries it) works across restarts."""
    path = LOG_DIR / ".token"
    try:
        tok = path.read_text().strip()
        if len(tok) == 32:
            return tok
    except OSError:
        pass
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_hex(16)
    # Created owner-only from the start (no window where it exists with default permissions).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
    os.chmod(path, 0o600)                                   # in case it existed with wider permissions
    return tok


TOKEN = _load_token()
# PokerNow's game pages (pokernow.club now redirects to pokernow.com; the live test ran on .com).
ALLOWED_ORIGINS = {"https://www.pokernow.com", "https://pokernow.com", "https://www.pokernow.club", "https://pokernow.club"}
PORT = {"n": 8765}
STATS = {"started": time.time(), "last_poll": 0.0, "decisions": 0, "acted": 0,
         "act_ms": collections.deque(maxlen=5000), "slow": 0,
         "errors": 0, "hand": 0, "last_action": "", "last_error": ""}
DEMO_LOG = Path(__file__).with_name("demo_log.csv")
DEMO = {"step": 0}                 # how many log lines of the demo hand are visible
DEMO_PAGE = """<!doctype html><meta charset="utf-8"><title>Demo table</title>
<body style="background:#0b3d2e;color:#fff;font:14px system-ui;padding:24px">
<h3>Demo table (not PokerNow)</h3>
<p>Replays demo_log.csv. "Next log line" plays the opponent; when it's the hero's turn the page marks it like
PokerNow does, and auto-play answers with these buttons. Each answer's time from the start of the turn is listed.</p>
<button id="next">Next log line</button> <button id="reset">Restart</button> <button id="auto">Auto-advance: off</button>
<span id="n"></span>
<div class="table-player you-player" id="you" style="margin:16px 0"><span class="table-player-name">Allen</span>
  <b id="turn"></b></div>
<div class="action-buttons"><button>Fold</button> <button>Check</button> <button>Call</button> <button>Raise</button></div>
<form class="raise-controller-form" style="display:none;margin-top:8px" onsubmit="return false">
  <input class="value" type="text"> <button type="button">All In</button> <input type="submit" value="Raise"></form>
<ol id="times"></ol>
<script>
const you = document.getElementById("you"), form = document.querySelector(".raise-controller-form");
let t0 = 0, auto = false;
const show = (j) => {
  document.getElementById("n").textContent = "line " + j.step;
  const on = j.hero_turn;
  if (on && !you.classList.contains("decision-current")) { you.classList.add("decision-current"); t0 = performance.now(); }
  if (!on) you.classList.remove("decision-current");
  document.getElementById("turn").textContent = on ? "  ← hero to act" : "";
};
const go = (u) => fetch(u, {method: "POST", headers: {"X-Pnbot-Token": "__PNBOT_TOKEN__"}}).then(r => r.json()).then(show);
const answered = (what) => {
  if (!you.classList.contains("decision-current")) return;
  const li = document.createElement("li");
  li.textContent = what + " at " + ((performance.now() - t0) / 1000).toFixed(3) + " s";
  document.getElementById("times").appendChild(li);
  form.style.display = "none";
  go("/games/demo/next");                          // the hero's own line in the scripted log
};
document.querySelectorAll(".action-buttons button").forEach(b => b.onclick = () =>
  b.textContent === "Raise" ? (form.style.display = "block") : answered(b.textContent));
form.querySelector("input[type=submit]").onclick = () => answered("Raise to " + form.querySelector("input.value").value);
document.getElementById("next").onclick = () => go("/games/demo/next");
document.getElementById("reset").onclick = () => { document.getElementById("times").innerHTML = ""; go("/games/demo/reset"); };
document.getElementById("auto").onclick = (e) => { auto = !auto; e.target.textContent = "Auto-advance: " + (auto ? "on" : "off"); };
// Driven by a worker so it keeps running in a hidden tab (like the bot's own clock).
const w = new Worker(URL.createObjectURL(new Blob(["setInterval(() => postMessage(0), 250)"], {type: "text/javascript"})));
let n = 0;
w.onmessage = () => { if (++n % 3) return go("/games/demo/state");
  if (auto && !you.classList.contains("decision-current")) go("/games/demo/next"); else go("/games/demo/state"); };
</script><script src="/client.js?k=__PNBOT_TOKEN__"></script></body>"""


def decide(payload):
    STATS["last_poll"] = time.time()
    msgs = read_log(payload["log"], payload.get("kind", "auto"))
    h = parse_hand(msgs, hero_seat=payload.get("seat"), hero_name=payload.get("you_name"))
    _track(msgs, h)
    table = describe(h)
    STATS["hand"] = h.number
    table["seats_in_hand"] = [{"seat": s.seat, "name": s.name} for s in h.seats.values()]
    if not h.seats:
        return {"table": table, "decision": None, "wait": "no hand in the log yet"}
    if h.hero() is None:
        return {"table": table, "decision": None, "wait": "hero seat not dealt in this hand"}
    if not h.heads_up():
        return {"table": table, "decision": None, "wait": f"{len(h.seats)} players dealt in: the bot is heads-up only"}
    if table["to_act"] != "hero":
        return {"table": table, "decision": None,
                "wait": "hand over" if h.over else "opponent to act" if table["to_act"] == "villain" else "waiting"}
    if len(h.hole) != 2:
        return {"table": table, "decision": None, "wait": "hero cards not in the log yet"}
    # Identify the decision point by what happened in the hand (its id line, the actions and cards), not by
    # the raw log text: chat or join lines must never make the same decision look new (a double action).
    start = max(k for k, m in enumerate(msgs) if m.startswith("-- starting hand"))
    point = repr((msgs[start], [[(a.player, a.kind, a.to) for a in st] for st in h.history], h.board, h.hole,
                  h.bb, h.sb, h.street, h.street_high, h.street_inc,
                  sorted((x.seat, x.start, round(x.put_in, 2), round(x.street_bet, 2), x.folded, x.all_in)
                         for x in h.seats.values())))
    digest = hashlib.sha256(point.encode()).hexdigest()[:16]
    key = (h.number, digest, h.hole[0] + h.hole[1], sum(len(x) for x in h.history), len(h.board))
    if key not in CACHE:
        t0 = time.perf_counter()
        hero = h.hero()
        opp = next((x for x in h.seats.values() if x is not hero), None)
        with DECIDE_LOCK:                                   # several tables: one opponent's state at a time
            ADAPT.apply(opp.pid if opp else None)
            d = ENGINE.decide(h.position(hero), "".join(h.hole), "".join(h.board), h.history, stack=h.effective_bb())
        to_chips, all_in = round(d.to * h.bb, 2), False
        if d.action in ("bet", "raise"):
            # PokerNow's rules: at least a min-raise, at most the stack; a size at the stack is an all-in.
            lo, hi = h.legal_to(hero)
            to_chips = max(to_chips, lo)
            if to_chips >= hi - 0.005:
                to_chips, all_in = hi, True
        elif d.action == "call":
            all_in = (table.get("to_call") or 0) >= h.stack_now(hero) - 0.005
        CACHE[key] = {"action": d.action, "to_chips": to_chips, "to_bb": round(to_chips / h.bb, 2), "all_in": all_in,
                      "label": d.label, "probs": {k: round(v, 4) for k, v in sorted(d.probs.items(), key=lambda kv: -kv[1])},
                      "node": d.node, "source": d.source, "notes": d.notes, "id": "-".join(map(str, key)),
                      "ms": round((time.perf_counter() - t0) * 1000, 1)}
        STATS["decisions"] += 1
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "decisions.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "table": table, "decision": CACHE[key]}) + "\n")
    return {"table": table, "decision": CACHE[key], "wait": None}


def _track(msgs, h):
    """Store every finished hand in the log page (bot/opponent.py) and refresh the adaptation. Never lets a
    tracking problem get in the way of a decision."""
    try:
        hero = h.hero()
        if hero is None:
            return
        for hand in split_hands(msgs):
            hid = hand[0]
            if hid in SEEN_HANDS:
                continue
            SEEN_HANDS.add(hid)
            rec = record(hand, hero.name)
            if rec and store([rec]):
                STATS["stored"] = STATS.get("stored", 0) + 1
            if rec:
                ADAPT.update(rec["opp_id"])
                x = results(load(rec["opp_id"]))
                print(f"[results] vs {rec['opp_name']} ({rec['opp_id']}): won {x['won']}/{x['hands']} hands "
                      f"({x['win_rate']:.0%}), net {x['net_bb']:+} bb ({x['bb_per_100']:+} bb/100), "
                      f"showdowns {x['showdowns_won']}/{x['showdowns']}", flush=True)
                if x["hands"] % 25 == 0:
                    rd = read_style(profile(load(rec["opp_id"])))
                    print(f"[read] {rec['opp_name']}: {', '.join(rd['style'])}"
                          f"{'; plays like a bot: ' + ', '.join(rd['bot_like']) if rd['bot_like'] else ''}; {rd['stats']}",
                          flush=True)
    except Exception as e:                                   # noqa: BLE001 - tracking is best effort
        STATS["track_error"] = repr(e)[:200]


THINK_MS = 4500                    # every action is committed exactly this long after the turn starts
TOLERANCE_MS = 250                 # an action further than this from the mark counts as off-target


def _clean(x, n):
    return re.sub(r"[\x00-\x1f\x7f]", " ", str(x))[:n] if x is not None else None


def acted(report):
    """The page reports each click: how long from seeing its turn to clicking, and whether it worked."""
    try:
        ms = float(report.get("ms", 0))
    except (TypeError, ValueError):
        ms = float("nan")
    if not math.isfinite(ms) or not 0 <= ms <= 3_600_000:
        raise ValueError("bad timing in action report")
    rec = {"id": _clean(report.get("id"), 64), "action": _clean(report.get("action"), 80), "ms": round(ms),
           "retry": bool(report.get("retry")), "error": _clean(report.get("error"), 200)}
    STATS["acted"] += 1
    if not rec["retry"]:                                     # retries are recovery, not timed actions
        STATS["act_ms"].append(ms)
        STATS["slow"] += abs(ms - THINK_MS) > TOLERANCE_MS
    STATS["last_action"] = f"hand {STATS['hand']}: {rec['action']} ({ms:.0f} ms)"
    if rec["error"]:
        STATS["errors"] += 1
        STATS["last_error"] = rec["error"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "actions.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), **rec}) + "\n")
    return {"ok": True}


def status_line():
    ms = sorted(STATS["act_ms"])
    p = (lambda q: ms[min(len(ms) - 1, int(q * len(ms)))]) if ms else (lambda q: 0)
    age = time.time() - STATS["last_poll"] if STATS["last_poll"] else float("inf")
    page = "page connected" if age < 10 else "NO PAGE POLLING (reload the PokerNow tab and run the bookmarklet)" \
        if STATS["last_poll"] else "waiting for the page"
    return (f"[{time.strftime('%H:%M:%S')}] hand {STATS['hand']} | actions {STATS['acted']} | time to act "
            f"median {p(0.5) / 1000:.2f}s, range {(ms[0] if ms else 0) / 1000:.2f}-{(ms[-1] if ms else 0) / 1000:.2f}s, "
            f"off the {THINK_MS / 1000:.2f}s mark by >{TOLERANCE_MS / 1000:.2f}s: {STATS['slow']} | "
            f"errors {STATS['errors']} | {page}" + (f" | last error: {STATS['last_error']}" if STATS["errors"] else "")
            + _adapt_line())


def _adapt_line():
    a = ADAPT.state
    if not a:
        return f" | hands stored {STATS.get('stored', 0)}" + (f" (tracking error {STATS['track_error']})"
                                                              if STATS.get("track_error") else "")
    p = a["postflop"]
    return (f" | opponent {a['hands']} hands, preflop adapted {a['lam']:.0%}, bluff x{p['bluff']:.2f}, "
            f"defend x{p['defend']:.2f}")


def monitor(every=30):
    while True:
        time.sleep(every)
        print(status_line(), flush=True)


def origin_ok(origin):
    """Requests must name an allowed origin; browsers always send one on these cross-site requests."""
    local = {f"http://127.0.0.1:{PORT['n']}", f"http://localhost:{PORT['n']}"}      # the demo page
    return origin is not None and (origin in local or origin in ALLOWED_ORIGINS)


class Handler(BaseHTTPRequestHandler):
    def _headers(self, code=200, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        origin = self.headers.get("Origin")
        if origin and origin_ok(origin):                    # CORS only for PokerNow (and the local demo)
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Pnbot-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")   # Chrome: public page -> localhost
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")      # the key in our URLs never leaks via Referer
        self.end_headers()

    def do_OPTIONS(self):
        self._headers(204 if origin_ok(self.headers.get("Origin")) else 403)

    def do_GET(self):
        route = self.path.split("?")[0]
        if route in ("/client.js", "/games/demo"):
            # Both carry the key, so both are served only to requests that already have it.
            query = dict(p.split("=", 1) for p in self.path.partition("?")[2].split("&") if "=" in p)
            if not secrets.compare_digest(query.get("k", ""), TOKEN):
                self._headers(403)
                return
        if route == "/client.js":
            self._headers(ctype="application/javascript; charset=utf-8")
            self.wfile.write(CLIENT.read_text().replace("__PNBOT_TOKEN__", TOKEN).encode())
        elif self.path.split("?")[0] == "/games/demo":
            self._headers(ctype="text/html; charset=utf-8")
            self.wfile.write(DEMO_PAGE.replace("__PNBOT_TOKEN__", TOKEN).encode())
        elif self.path.split("?")[0] == "/games/demo/poker_now_log_demo.csv":
            # PokerNow's CSV is newest first; show only the lines the demo has reached.
            import csv
            import io
            rows = list(csv.DictReader(io.StringIO(DEMO_LOG.read_text())))
            rows = sorted(rows, key=lambda r: int(r["order"]))[:DEMO["step"]]
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=["entry", "at", "order"])
            w.writeheader()
            w.writerows(reversed(rows))
            self._headers(ctype="text/csv; charset=utf-8")
            self.wfile.write(buf.getvalue().encode())
        elif self.path.split("?")[0] == "/ping":
            self._headers()
            self.wfile.write(b'{"ok": true}')
        else:
            self._headers(404)

    def do_POST(self):
        # Every POST needs an allowed Origin and the key (the demo page carries it too).
        if not origin_ok(self.headers.get("Origin")) or not secrets.compare_digest(
                self.headers.get("X-Pnbot-Token", ""), TOKEN):
            self._headers(403)
            self.wfile.write(b'{"error": "forbidden: reload the bot (click the bookmarklet again)"}')
            return
        if self.path in ("/games/demo/next", "/games/demo/reset", "/games/demo/state"):
            n = sum(1 for _ in DEMO_LOG.read_text().splitlines()) - 1
            DEMO["step"] = {"next": min(DEMO["step"] + 1, n), "reset": 0}.get(self.path.rsplit("/", 1)[1], DEMO["step"])
            msgs = read_log(DEMO_LOG.read_text())[:DEMO["step"]]
            hh = parse_hand(msgs, hero_seat=3) if msgs else None
            hero_turn = bool(hh) and describe(hh)["to_act"] == "hero" and len(hh.hole) == 2
            self._headers()
            self.wfile.write(json.dumps({"step": DEMO["step"], "hero_turn": hero_turn}).encode())
            return
        if self.path not in ("/decide", "/acted"):
            self._headers(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            out = decide(json.loads(body)) if self.path == "/decide" else acted(json.loads(body))
            code = 200
        except Exception as e:                              # report parse problems to the overlay
            STATS["errors"] += 1
            STATS["last_error"] = f"{type(e).__name__}: {e}"
            out, code = {"error": f"{type(e).__name__}: {e}"}, 500
        self._headers(code)
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, fmt, *args):                      # keep the terminal quiet
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    PORT["n"] = port
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno != 48 and "in use" not in str(e):
            raise
        sys.exit(f"Port {port} is already in use: another copy of this server is probably running (its overlay "
                 f"keeps working). Stop it first (lsof -ti :{port} | xargs kill), or pick another port: "
                 f"python -m bot.pokernow.server {port + 1}")
    bookmarklet = ("javascript:(()=>{const s=document.createElement('script');"
                   f"s.src='http://127.0.0.1:{port}/client.js?k={TOKEN}&t='+Date.now();document.body.appendChild(s)}})()")
    t0 = time.perf_counter()
    ENGINE.warm()                                           # no disk reads during the game
    print(f"poker bot server on http://127.0.0.1:{port}  (policies: {ENGINE.policy_dir.name}, "
          f"loaded in {time.perf_counter() - t0:.1f}s)")
    print("bookmarklet (save as a bookmark's URL, click it on the PokerNow game tab; it carries a private key,")
    print("so don't share it; the key stays the same across restarts, delete output/pokernow/.token to change it):")
    print(bookmarklet)
    print(f"demo table (to check the overlay without PokerNow): http://127.0.0.1:{port}/games/demo?k={TOKEN}")
    import threading
    threading.Thread(target=monitor, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
