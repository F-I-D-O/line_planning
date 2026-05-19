"""
Lightweight checks for DARP pool selection and export mapping (no Gurobi / instance load).

Run from repo root::

    python scripts/test_darp_pool_clustered_avg_sanity.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_mod_aware():
    path = Path(__file__).resolve().parent / "MoD-aware_line_selection.py"
    spec = importlib.util.spec_from_file_location("mod_aware_line_selection_sanity", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = _load_mod_aware()
    pool_df = mod._normalize_pool_df(
        pd.DataFrame(
            {
                "pool_id": [0, 1, 2],
                "passenger_idx": [0, 0, 0],
                "route_idx": [-1, 0, 0],
                "leg_kind": [
                    mod.DARP_POOL_LEG_NO_MT,
                    mod.DARP_POOL_LEG_FIRST_MILE,
                    mod.DARP_POOL_LEG_LAST_MILE,
                ],
                "origin": [10, 10, 40],
                "destination": [20, 30, 20],
                "time": [0.0, 0.0, 5.0],
            }
        )
    )

    class FakeLineInstance:
        nb_pass = 1

        @staticmethod
        def trip_option_position(passenger_idx, line_idx):
            if int(passenger_idx) == 0 and int(line_idx) == 0:
                return 0
            return None

    line_inst = FakeLineInstance()
    assert mod._pool_id_for_leg(line_inst, pool_df, 0, None, mod.DARP_POOL_LEG_NO_MT) == 0
    assert mod._pool_id_for_leg(line_inst, pool_df, 0, 0, mod.DARP_POOL_LEG_FIRST_MILE) == 1
    assert mod._pool_id_for_leg(line_inst, pool_df, 0, 0, mod.DARP_POOL_LEG_LAST_MILE) == 2

    assignments = [("line", 0)]
    reqs, exp_map = mod.select_darp_requests_from_pool(pool_df, assignments, line_inst)
    assert len(reqs) == 2
    assert exp_map[0] == 1 and exp_map[1] == 2
    assert reqs[0]["original_request_id"] == 0 and reqs[1]["original_request_id"] == 0

    assignments2 = [("no_MT", None)]
    reqs2, exp2 = mod.select_darp_requests_from_pool(pool_df, assignments2, line_inst)
    assert len(reqs2) == 1
    assert exp2[0] == 0

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "darp_pool_export_map.json"
        mod.write_darp_pool_export_map(p, exp_map)
        loaded = mod.load_darp_pool_export_map(p)
        assert loaded == exp_map

    print("darp pool clustered_avg sanity checks: OK")


if __name__ == "__main__":
    main()
