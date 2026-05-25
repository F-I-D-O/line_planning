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

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lineplanning.instance import line_instance as LineInstance


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

    lookup_inst = LineInstance.__new__(LineInstance)
    lookup_inst.nb_lines = 4
    lookup_inst._line_position_cache = {}
    lookup_inst.optimal_trip_options = pd.DataFrame(
        {
            "passenger_idx": np.asarray([1, 0, 0], dtype=np.int32),
            "line_idx": np.asarray([0, 2, 1], dtype=np.int32),
        }
    )
    lookup_inst.optimal_trip_options = lookup_inst._sort_trip_options_for_lookup(
        lookup_inst.optimal_trip_options
    )
    lookup_inst._rebuild_trip_option_index()
    assert lookup_inst.trip_option_position(0, 1) == 0
    assert lookup_inst.trip_option_position(0, 2) == 1
    assert lookup_inst.has_trip_option_on_line(1, 0)
    assert lookup_inst.trip_options_for_passenger(0)["line_idx"].tolist() == [1, 2]

    class FakePoolLineInstance(FakeLineInstance):
        nb_pass = 1
        nb_lines = 1
        requests = np.asarray([[10, 20]], dtype=np.int32)
        demand = pd.DataFrame(
            {
                "origin": np.asarray([10], dtype=np.int32),
                "destination": np.asarray([20], dtype=np.int32),
                "time": np.asarray([12.5], dtype=np.float32),
            }
        )
        dm = np.zeros((50, 50), dtype=np.float32)
        dm[10, 30] = 4.0
        lengths_travel_times = [10.0]
        optimal_trip_options = pd.DataFrame(
            {
                "passenger_idx": np.asarray([0], dtype=np.int32),
                "line_idx": np.asarray([0], dtype=np.int32),
                "mt_pickup_node": np.asarray([30], dtype=np.int32),
                "mt_drop_off_node": np.asarray([40], dtype=np.int32),
                "mt_pickup_line_edge_index": np.asarray([0], dtype=np.int16),
                "mt_drop_off_line_edge_index": np.asarray([2], dtype=np.int16),
            }
        )

        @staticmethod
        def line_length(line_idx):
            return 4

    built_pool = mod.build_darp_request_pool(FakePoolLineInstance(), transfer_delay=1)
    assert built_pool["time"].tolist() == [12.5, 12.5, 22.5]

    assignments = [("line", 0)]
    reqs, exp_map = mod.select_darp_requests_from_pool(pool_df, assignments, line_inst)
    assert len(reqs) == 2
    assert exp_map[0] == 1 and exp_map[1] == 2
    assert reqs[0]["original_request_id"] == 0 and reqs[1]["original_request_id"] == 0

    assignments2 = [("no_MT", None)]
    reqs2, exp2 = mod.select_darp_requests_from_pool(pool_df, assignments2, line_inst)
    assert len(reqs2) == 1
    assert exp2[0] == 0

    node_id_to_latlng = {
        10: (40.0, -73.0),
        20: (40.0, -73.0),
        30: (40.01, -73.01),
        40: (40.02, -73.02),
    }
    labels = mod.fit_hex_od_pool_cluster_labels(pool_df, node_id_to_latlng, 8)
    labels_again = mod.fit_hex_od_pool_cluster_labels(pool_df, node_id_to_latlng, 8)
    assert labels.dtype == "int32"
    assert labels.tolist() == labels_again.tolist()

    duplicate_pool_df = mod._normalize_pool_df(
        pd.concat([pool_df, pool_df.iloc[[1]].assign(pool_id=3)], ignore_index=True)
    )
    duplicate_labels = mod.fit_hex_od_pool_cluster_labels(duplicate_pool_df, node_id_to_latlng, 8)
    assert duplicate_labels[1] == duplicate_labels[3]
    assert len(set(int(x) for x in duplicate_labels)) >= 2

    missing_pool_df = mod._normalize_pool_df(pool_df.assign(origin=[999, 10, 40]))
    try:
        mod.fit_hex_od_pool_cluster_labels(missing_pool_df, node_id_to_latlng, 8)
    except ValueError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError("missing node id should raise ValueError")

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "darp_pool_export_map.json"
        mod.write_darp_pool_export_map(p, exp_map)
        loaded = mod.load_darp_pool_export_map(p)
        assert loaded == exp_map

    print("darp pool clustered_avg sanity checks: OK")


if __name__ == "__main__":
    main()
