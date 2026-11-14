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


def read_token(path=TOKEN_FILE):
    """The token from a private file: no symlinks, owned by this user, no group/other permissions."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return ""
    with os.fdopen(fd) as f:
        st = os.fstat(f.fileno())
        if st.st_uid != os.getuid() or st.st_mode & 0o077:
            sys.exit(f"refusing {path}: it must be yours and private (chmod 600)")
        return f.read().strip()


def private_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def main(argv):
    from chipzen import run_external_bot

    if len(argv) != 1:
        sys.exit(__doc__)
    token = os.environ.get("CHIPZEN_TOKEN") or read_token()
    if not token.startswith("cz_extbot_"):
        sys.exit(f"no token: put it in {TOKEN_FILE} (chmod 600) or $CHIPZEN_TOKEN")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run_external_bot(lambda: make_bot(log_dir=private_dir(ROOT / "output" / "chipzen")), bot_id=argv[0], env="prod",
                                 token=token, client_name="hubot", user_agent="hubot/8"))


if __name__ == "__main__":
    main(sys.argv[1:])
