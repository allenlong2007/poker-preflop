"""Exact-card, action-time evidence from revealed hands. Never an inference-time feature source."""
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import eval7

from preflop.exactgame import DECK, SEATS, cards


def card_features(hole, board):
    combined = hole + board
    suits = Counter(c[1] for c in combined)
    ranks = Counter(c[0] for c in combined)
    score = eval7.evaluate([eval7.Card(c) for c in combined]) if board else None
    flush_draw = [s for s in "cdhs" if suits[s] == 4 and any(c[1] == s for c in hole)] if len(board) < 5 else []
    straight_outs = []
    if 3 <= len(board) < 5 and (score >> 24) not in (4, 8):
        for card in DECK:
            if card not in combined:
                future = eval7.evaluate([eval7.Card(c) for c in combined + (card,)])
                if future >> 24 in (4, 8):
                    straight_outs.append(card)
    return {"evaluation": score, "category": eval7.handtype(score) if score is not None else None,
            "rank_counts": dict(ranks), "suit_counts": dict(suits),
            "hole_ranks": [c[0] for c in hole], "hole_suits": [c[1] for c in hole],
            "flush_draw_suits": flush_draw,
            "flush_completing_cards": [c for c in DECK if c not in combined and c[1] in flush_draw],
            "straight_completing_cards": straight_outs}


def analyze_revealed(record):
    revealed = record.get("shown", {}).get("opp")
    if not revealed:
        return None
    hole, full_board = cards(revealed), cards(record.get("board", ""))
    hero = cards(record.get("hero_hole", ""))
    if len(hole) != 2 or set(hole) & (set(full_board) | set(hero)):
        raise ValueError("revealed cards conflict with known cards")
    actions, total = [], {"btn": 0.0, "bb": 0.0}
    prefix = []
    for street, history in enumerate(record["history"]):
        board = full_board[:(0, 3, 4, 5)[street]]
        if street and len(board) != street + 2:
            break
        put = {"btn": 0.5, "bb": 1.0} if street == 0 else {"btn": 0.0, "bb": 0.0}
        current = []
        for player, kind, to in history:
            pot = sum(total.values()) + sum(put.values())
            facing = max(put.values()) - put[player]
            if player == record["opp_pos"]:
                actions.append({"street": street, "action": kind, "to_bb": to,
                                "board": list(board), "history_before": prefix + [list(current)],
                                "pot_before_bb": pot, "facing_bb": facing,
                                "price": facing / (pot + facing) if facing else 0.0,
                                "features": card_features(hole, board)})
            current.append([player, kind, to])
            if kind in ("bet", "raise"):
                put[player] = to
            elif kind == "call":
                put[player] = min(max(put.values()), record["stacks"]["hero" if player == record["hero_pos"] else "opp"] - total[player])
        for player in SEATS:
            total[player] += put[player]
        prefix.append(current)
    return {"schema_version": 1, "selection": "revealed_hands_only", "available_after": "hand_finished",
            "opponent_hole": list(hole), "actions": actions}


class EvidenceIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS reveals (id TEXT PRIMARY KEY, opponent TEXT NOT NULL, "
                        "combo TEXT NOT NULL, evidence TEXT NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS reveals_opponent ON reveals(opponent)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def add(self, record):
        evidence = analyze_revealed(record)
        if evidence is None:
            return
        identity = json.dumps([record.get("hero_id"), record["id"], record["opp_id"]])
        combo = "".join(sorted(evidence["opponent_hole"], key=DECK.index, reverse=True))
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO reveals VALUES (?,?,?,?)",
                            (identity, record["opp_id"], combo, json.dumps(evidence, allow_nan=False)))

    def summary(self, opponent=None):
        where, args = (" WHERE opponent=?", (opponent,)) if opponent else ("", ())
        rows = self.db.execute("SELECT combo,evidence FROM reveals" + where, args).fetchall()
        counts = Counter()
        for _, evidence in rows:
            counts.update(f"{a['street']}:{a['action']}" for a in json.loads(evidence)["actions"])
        return {"selection": "revealed_hands_only", "revealed_hands": len(rows),
                "exact_combos": dict(Counter(combo for combo, _ in rows)), "actions": dict(counts)}


def main():
    from bot.opponent import HANDS, load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    rejected = 0
    with EvidenceIndex(HANDS.parent / "exact_evidence.sqlite") as index:
        if args.rebuild:
            for record in load(args.opponent):
                try:
                    index.add(record)
                except (ValueError, KeyError) as error:
                    rejected += 1
                    print(f"Invalid revealed record {record.get('id')}: {error}")
        print(json.dumps({**index.summary(args.opponent), "invalid_records": rejected}, indent=2))


if __name__ == "__main__":
    main()
