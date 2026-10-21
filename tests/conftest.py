import pytest


@pytest.fixture(autouse=True)
def _private_hand_store(tmp_path, monkeypatch):
    """Tests must never write into the real opponent hand store (output/pokernow/hands)."""
    from bot import opponent
    monkeypatch.setattr(opponent, "HANDS", tmp_path / "hands")
