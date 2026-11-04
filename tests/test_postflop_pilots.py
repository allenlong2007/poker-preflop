import json
import math

import numpy as np
import pytest

from bot.engine import Act, Engine, State, parse_cards


@pytest.mark.parametrize("stack, hi_weight", [(30, 0), (40, 0), (60, math.log(1.5) / math.log(2.5)), (100, 1)])
def test_depth_model_blends_postflop_rows(tmp_path, stack, hi_weight):
    engine = Engine(policy_dir=tmp_path / "policy_fine", exact_required=False, use_solver_flops=False,
                    postflop_depths=[40, 100])
    sizes = {"donk": True, "sizes": [{"lead": [1 / 3], "bet": [1 / 3]}] * 3}
    shape = (100, 1, 12, 2)
    engine._pots["srp_alt"] = {"pot_to": 3}
    engine._pots["srp"] = {
        "opts_json": json.dumps(sizes),
        "sigma_0_bb_first": np.broadcast_to([0.75, 0.25], shape),
    }
    folder = tmp_path / "policy_depth" / "40"
    folder.mkdir(parents=True)
    np.savez(folder / "srp.npz", opts_json=json.dumps(sizes),
             sigma_0_bb_first=np.broadcast_to([0.25, 0.75], shape))
    state = State([[Act("btn", "raise", 2.25), Act("bb", "call")]], stack)
    decision = engine._postflop("bb", parse_cards("AcQd"), parse_cards("Kh8d3c"), state)
    assert decision.diagnostics["base"]["probs"]["check"] == pytest.approx(0.25 + 0.5 * hi_weight)
    assert sum(decision.diagnostics["lookup"]["postflop_depths"].values()) == pytest.approx(1)
