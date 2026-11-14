"""Run HUBot v8 on Chipzen.ai as a remote (External API) bot. The code never leaves this Mac.

Setup, once:
  1. On chipzen.ai: My Bots > Create bot > External API. Copy the bot id (a UUID).
  2. On the bot page: API tokens > Create token. Save it to output/chipzen/token (git-ignored), then
     `chmod 600 output/chipzen/token`. It is shown once; rotate it on the site if it leaks.

Run (keep the Mac awake and on AC for every fixture):
  caffeinate -is .venv/bin/python scripts/chipzen_remote.py <bot id>

Decisions are logged to output/chipzen/decisions.jsonl. The token is never printed.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.chipzen.adapter import make_bot  # noqa: E402

TOKEN_FILE = ROOT / "output" / "chipzen" / "token"


def main(argv):
    from chipzen import run_external_bot

    if len(argv) != 1:
        sys.exit(__doc__)
    token = os.environ.get("CHIPZEN_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not token.startswith("cz_extbot_"):
        sys.exit(f"no token: put it in {TOKEN_FILE} (chmod 600) or $CHIPZEN_TOKEN")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run_external_bot(lambda: make_bot(log_dir=ROOT / "output" / "chipzen"), bot_id=argv[0], env="prod",
                                 token=token, client_name="hubot", user_agent="hubot/8"))


if __name__ == "__main__":
    main(sys.argv[1:])
