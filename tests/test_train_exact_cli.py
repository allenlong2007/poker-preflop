"""Bounded CPU training and frozen evaluation can be driven without a live table."""
import json

import pytest

from scripts.train_exact import main
from preflop.exactgame import Config
from preflop.exacttrain import Trainer


def test_training_status_snapshot_and_rejected_promotion(tmp_path, capsys):
    candidate = tmp_path / "candidate.sqlite"
    frozen = tmp_path / "frozen.sqlite"
    report = tmp_path / "evaluation.json"
    main(["train", "--path", str(candidate), "--iterations", "4", "--batch", "2",
          "--chip-bb", "0.5", "--stacks", "3", "--max-gb", "0.1"])
    main(["status", "--path", str(candidate)])
    assert '"iterations": 4' in capsys.readouterr().out
    main(["snapshot", "--path", str(candidate), "--output", str(frozen)])
    main(["evaluate", "--path", str(frozen), "--deals", "2", "--min-updates", "1", "--output", str(report)])
    metrics = json.loads(report.read_text())
    assert metrics["deals"] == 2 and metrics["illegal"] == 0
    assert metrics["candidate_sha256"]
    assert metrics["promotion_errors"]
    with pytest.raises(ValueError, match="promotion"):
        main(["promote", "--path", str(frozen), "--report", str(report), "--output", str(tmp_path / "active.sqlite")])
    assert not (tmp_path / "active.sqlite").exists()


def test_training_requires_a_finite_budget(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        main(["train", "--path", str(tmp_path / "never_created.sqlite")])


def test_snapshot_reads_a_consistent_checkpoint_while_writer_is_open(tmp_path):
    path, output = tmp_path / "writer.sqlite", tmp_path / "snapshot.sqlite"
    with Trainer(path, Config(chip_bb=0.5, stacks_bb=(3,))) as trainer:
        trainer.train(3)
        main(["snapshot", "--path", str(path), "--output", str(output)])
        trainer.train(2)
    with Trainer(output) as snapshot:
        assert snapshot.status()["iterations"] == 3


def test_evaluation_rejects_a_mutable_wal_candidate(tmp_path):
    path = tmp_path / "writer.sqlite"
    with Trainer(path, Config(chip_bb=0.5, stacks_bb=(3,))) as trainer:
        trainer.train(1)
        with pytest.raises(ValueError, match="frozen"):
            main(["evaluate", "--path", str(path), "--deals", "2", "--output", str(tmp_path / "bad.json")])
