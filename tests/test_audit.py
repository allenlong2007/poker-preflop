"""Durable decision provenance, including restart and old-history attribution."""
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from bot.audit import AuditLog


@pytest.fixture
def audit_engine(tmp_path):
    root = tmp_path / "project"
    (root / "bot").mkdir(parents=True)
    (root / "bot" / "version.py").write_text("VERSION = 6\n")
    policy = root / "output" / "match" / "policy"
    policy.mkdir(parents=True)
    (policy / "preflop.json").write_text("{}")
    (policy / "srp.npz").write_bytes(b"policy one")
    flops = root / "flops"
    flops.mkdir()
    (flops / "index.json").write_text("{}")
    engine = SimpleNamespace(policy_dir=policy, flops_dir=flops, use_solver_flops=True,
                             translate=True, pot_odds=True)
    return root, engine


def decision_row(hand="hand1", hero="hero1", decision="decision1"):
    return {"hand_id": hand, "hero_id": hero, "decision": {"id": decision, "action": "call"}}


def test_build_fingerprints_detect_source_and_policy_changes(tmp_path, audit_engine):
    root, engine = audit_engine
    first = AuditLog(tmp_path / "logs", engine, root=root)
    same = AuditLog(tmp_path / "logs", engine, root=root)
    assert first.run["run_id"] != same.run["run_id"]
    assert first.run["source_sha256"] == same.run["source_sha256"]
    assert first.run["policy_sha256"] == same.run["policy_sha256"]
    (root / "bot" / "version.py").write_text("VERSION = 7\n")
    (engine.policy_dir / "srp.npz").write_bytes(b"policy two")
    changed = AuditLog(tmp_path / "logs", engine, root=root)
    assert changed.run["source_sha256"] != first.run["source_sha256"]
    assert changed.run["policy_sha256"] != first.run["policy_sha256"]
    assert changed.run["git_commit"] is None  # a source export without .git works


def test_decisions_have_run_metadata_and_one_manifest(tmp_path, audit_engine):
    root, engine = audit_engine
    audit = AuditLog(tmp_path / "logs", engine, root=root)
    audit.write_decision(decision_row())
    audit.write_decision(decision_row(decision="decision2"))
    rows = [json.loads(line) for line in (audit.directory / "decisions.jsonl").read_text().splitlines()]
    runs = [json.loads(line) for line in (audit.directory / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 1
    assert rows[0]["schema_version"] == 1
    assert rows[0]["run_id"] == runs[0]["run_id"]
    assert rows[0]["bot_version"] == runs[0]["bot_version"] == 6
    assert runs[0]["config"]["pot_odds"] is True
    assert audit.hand_metadata("hand1", "hero1")["decision_ids"] == ["decision1", "decision2"]


def test_restart_preserves_original_run_and_leaves_unseen_hands_unknown(tmp_path, audit_engine):
    root, engine = audit_engine
    first = AuditLog(tmp_path / "logs", engine, root=root)
    first.write_decision(decision_row())
    second = AuditLog(first.directory, engine, root=root)
    assert second.hand_metadata("hand1", "hero1")["run_ids"] == [first.run["run_id"]]
    assert second.hand_metadata("hand1", "other_hero") is None
    assert second.hand_metadata("old_imported_hand", "hero1") is None
    second.write_decision(decision_row(decision="decision2"))
    assert second.hand_metadata("hand1", "hero1")["run_ids"] == [first.run["run_id"], second.run["run_id"]]


def test_recovery_tolerates_legacy_and_truncated_records(tmp_path, audit_engine):
    root, engine = audit_engine
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "decisions.jsonl").write_text('{"t": 1, "decision": {}}\n{"incomplete":')
    audit = AuditLog(directory, engine, root=root)
    audit.write_decision(decision_row())
    restarted = AuditLog(directory, engine, root=root)
    assert restarted.hand_metadata("hand1", "hero1")["decision_ids"] == ["decision1"]


def test_server_persists_context_and_links_only_observed_hands(tmp_path, monkeypatch):
    from bot.opponent import record, split_hands
    from bot.pokernow import server

    demo = Path(__file__).resolve().parents[1] / "bot" / "pokernow" / "demo_log.csv"
    msgs = server.read_log(demo.read_text())
    end = next(i for i, msg in enumerate(msgs) if "bets 4.10" in msg)

    monkeypatch.setattr(server, "LOG_DIR", tmp_path)
    monkeypatch.setattr(server, "CACHE", {})
    monkeypatch.setattr(server, "SEEN_HANDS", set())
    monkeypatch.setattr(server, "_track", lambda *args: None)
    monkeypatch.setattr(server.ADAPT, "apply", lambda *args: None)
    payload = {"log": json.dumps(msgs[:end + 1]), "seat": 3}
    first = server.decide(payload)
    assert first == server.decide(payload)
    rows = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["hand_id"] == "h2abc" and row["hero_id"] == "aL1"
    assert row["opponent_id"] == "fR7"
    assert row["context"]["hero_hole"] == "KsQs"
    assert row["context"]["history"][0][0] == ["btn", "raise", 3.0]
    assert row["decision"]["run_id"] == row["run_id"]
    assert row["decision"]["diagnostics"]["pot_odds"]["price"] > 0
    assert row["decision"]["diagnostics"]["adjustments"]
    assert row["context"]["adaptation"]["strictness"] == server.ENGINE.strict
    assert record(split_hands(msgs)[1], hero_name="Allen")["bot_version"] is None
    stored = []
    monkeypatch.setattr(server, "store", lambda rows: stored.extend(rows) or len(rows))
    monkeypatch.setattr(server.ADAPT, "update", lambda *args: None)
    monkeypatch.setattr(server, "load", lambda *args: stored)
    server._track_locked(msgs, server.parse_hand(msgs, hero_seat=3))
    assert len(stored) == 2
    assert stored[0]["bot_version"] is None
    assert stored[1]["bot_version"] == 6
    assert stored[1]["audit"]["decision_ids"] == [row["decision"]["id"]]
