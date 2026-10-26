"""Losslessly migrate a schema-1 exact MCCFR artifact to schema 2, verifying every node.

Usage:
    python scripts/migrate_exact_storage.py SOURCE DEST [--source-commit 76d16af] [--key-text-floor 100]

The source is read-only and is never modified or deleted; DEST appears only after verification passes. A JSON report
is printed and written to DEST.migration.json. See docs/storage-bounded-continuation.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflop.exactmigrate import migrate
from preflop.exactstore import KEY_TEXT_FLOOR


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-commit", help="git commit whose training code produced SOURCE (checked)")
    parser.add_argument("--key-text-floor", type=int, default=KEY_TEXT_FLOOR)
    args = parser.parse_args(argv)
    started = time.time()

    def progress(stage, rows):
        print(f"[{time.time() - started:7.0f}s] {stage} {rows:,} rows", flush=True)

    report = migrate(args.source, args.destination, args.key_text_floor, args.source_commit, progress)
    Path(str(args.destination) + ".migration.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
