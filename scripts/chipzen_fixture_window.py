"""Keep the Chipzen remote runner connected only around our season fixtures, to save the daily lobby-time allowance.

usage: chipzen_fixture_window.py <competition_id> <bot_id>

Every minute: read the public fixtures list. If none of them include our bot yet (fixtures not published), leave the
runner alone. Once they are published, the runner should be up from LEAD before each unfinished fixture until TAIL
after it (or while a match is still writing hands), and down otherwise. Logs to output/chipzen/window.log.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/Users/allenlong/poker-preflop/.venv/bin/python"
LOG = ROOT / "output/chipzen/window.log"
RUNLOG = ROOT / "output/chipzen/runner.log"
HANDS = ROOT / "output/chipzen/hands.jsonl"
LEAD = timedelta(minutes=20)
TAIL = timedelta(minutes=75)
ACTIVE_S = 240            # hands written this recently = a match is in progress, never stop
DONE = {"completed", "walkover", "cancelled", "expired", "forfeit"}


def log(msg):
    with LOG.open("a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")


def fixtures(comp):
    # curl, not urllib: this python.org build has no CA bundle installed
    out = subprocess.run(["curl", "-sf", "-m", "20", f"https://chipzen.ai/api/competitions/{comp}/fixtures"],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    return d.get("fixtures", d) if isinstance(d, dict) else d


def runner_pids(bot):
    out = subprocess.run(["pgrep", "-f", f"chipzen_remote.py {bot}"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def start(bot):
    with RUNLOG.open("a") as f:
        subprocess.Popen(["caffeinate", "-is", PY, "scripts/chipzen_remote.py", bot], cwd=ROOT, stdout=f, stderr=f,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    log("runner START")


def stop(bot):
    subprocess.run(["pkill", "-f", f"caffeinate -is .*chipzen_remote.py {bot}"])
    for p in runner_pids(bot):
        subprocess.run(["kill", str(p)])
    log("runner STOP")


def main(comp, bot):
    log(f"manager up: competition {comp}, lead {LEAD}, tail {TAIL}")
    published, last_want = False, None
    while True:
        try:
            fx = [f for f in fixtures(comp) if bot in (f.get("bot_a_id"), f.get("bot_b_id"))]
        except Exception as e:                       # network blip: keep the current state
            log(f"fetch failed: {e!r}")
            time.sleep(60)
            continue
        if not fx:
            time.sleep(300 if not published else 60)
            continue
        if not published:
            published = True
            for f in sorted(fx, key=lambda f: f["scheduled_at"]):
                opp = f["bot_b_name"] if f["bot_a_id"] == bot else f["bot_a_name"]
                log(f"fixture {f['scheduled_at']} vs {opp} ({f.get('state')})")
        now = datetime.now(timezone.utc)
        want = any(f.get("state") not in DONE and
                   datetime.fromisoformat(f["scheduled_at"].replace("Z", "+00:00")) - LEAD <= now <=
                   datetime.fromisoformat(f["scheduled_at"].replace("Z", "+00:00")) + TAIL for f in fx)
        active = HANDS.exists() and time.time() - HANDS.stat().st_mtime < ACTIVE_S
        up = bool(runner_pids(bot))
        if (want or active) and not up:
            start(bot)
        elif not want and not active and up:
            stop(bot)
        if want != last_want:
            log(f"window {'OPEN' if want else 'closed'}")
            last_want = want
        time.sleep(60)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
