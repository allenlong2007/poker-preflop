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
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bot.engine import Engine
from bot.pokernow.parse import describe, parse_hand, read_log

ROOT = Path(__file__).resolve().parents[2]
CLIENT = Path(__file__).with_name("client.js")
LOG_DIR = ROOT / "output" / "pokernow"
ENGINE = Engine()
CACHE = {}                         # (hand number, actions so far) -> decision, so polling doesn't resample
DEMO_LOG = Path(__file__).with_name("demo_log.csv")
DEMO = {"step": 0}                 # how many log lines of the demo hand are visible
DEMO_PAGE = """<!doctype html><meta charset="utf-8"><title>Demo table</title>
<body style="background:#0b3d2e;color:#fff;font:14px system-ui;padding:24px">
<h3>Demo table (not PokerNow)</h3><p>Replays demo_log.csv one line per click, as if it were a live game.</p>
<button id="next">Next log line</button> <button id="reset">Restart</button> <span id="n"></span>
<div class="table-player you-player" style="margin-top:16px"><span class="table-player-name">Allen</span></div>
<script>
const go = (u) => fetch(u, {method: "POST"}).then(r => r.json()).then(j => document.getElementById("n").textContent = "line " + j.step);
document.getElementById("next").onclick = () => go("/games/demo/next");
document.getElementById("reset").onclick = () => go("/games/demo/reset");
go("/games/demo/state");
</script><script src="/client.js"></script></body>"""


def decide(payload):
    msgs = read_log(payload["log"], payload.get("kind", "auto"))
    h = parse_hand(msgs, hero_seat=payload.get("seat"), hero_name=payload.get("you_name"))
    table = describe(h)
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
    key = (h.number, h.hole[0] + h.hole[1], sum(len(x) for x in h.history), len(h.board))
    if key not in CACHE:
        pos = h.position(h.hero())
        d = ENGINE.decide(pos, "".join(h.hole), "".join(h.board), h.history, stack=h.effective_bb())
        to_chips = round(d.to * h.bb, 2)
        CACHE[key] = {"action": d.action, "to_chips": to_chips, "to_bb": round(d.to, 2), "label": d.label,
                      "probs": {k: round(v, 4) for k, v in sorted(d.probs.items(), key=lambda kv: -kv[1])},
                      "node": d.node, "source": d.source, "notes": d.notes, "id": "-".join(map(str, key))}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "decisions.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "table": table, "decision": CACHE[key]}) + "\n")
    return {"table": table, "decision": CACHE[key], "wait": None}


class Handler(BaseHTTPRequestHandler):
    def _headers(self, code=200, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")   # Chrome: public page -> localhost
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_OPTIONS(self):
        self._headers(204)

    def do_GET(self):
        if self.path.split("?")[0] == "/client.js":
            self._headers(ctype="application/javascript; charset=utf-8")
            self.wfile.write(CLIENT.read_bytes())
        elif self.path.split("?")[0] == "/games/demo":
            self._headers(ctype="text/html; charset=utf-8")
            self.wfile.write(DEMO_PAGE.encode())
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
        if self.path.startswith("/games/demo/"):
            n = sum(1 for _ in DEMO_LOG.read_text().splitlines()) - 1
            DEMO["step"] = {"next": min(DEMO["step"] + 1, n), "reset": 0}.get(self.path.rsplit("/", 1)[1], DEMO["step"])
            self._headers()
            self.wfile.write(json.dumps({"step": DEMO["step"]}).encode())
            return
        if self.path != "/decide":
            self._headers(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            out = decide(json.loads(body))
            code = 200
        except Exception as e:                              # report parse problems to the overlay
            out, code = {"error": f"{type(e).__name__}: {e}"}, 500
        self._headers(code)
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, fmt, *args):                      # keep the terminal quiet
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    bookmarklet = ("javascript:(()=>{const s=document.createElement('script');"
                   f"s.src='http://127.0.0.1:{port}/client.js?'+Date.now();document.body.appendChild(s)}})()")
    print(f"poker bot server on http://127.0.0.1:{port}  (policies: {ENGINE.policy_dir.name})")
    print("bookmarklet (save as a bookmark's URL, click it on the PokerNow game tab):")
    print(bookmarklet)
    srv.serve_forever()


if __name__ == "__main__":
    main()
