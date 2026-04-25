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
    rows = [
        mod.DarpPoolRow(0, 0, None, "no_mt", 10, 20, 0.0),
        mod.DarpPoolRow(1, 0, 0, "first_mile", 10, 30, 0.0),
        mod.DarpPoolRow(2, 0, 0, "last_mile", 40, 20, 5.0),
    ]
    m = mod.pool_key_to_pool_id_map(rows)
    assert m[(0, None, "no_mt")] == 0
    assert m[(0, 0, "first_mile")] == 1
    assert m[(0, 0, "last_mile")] == 2

    assignments = [("line", 0)]
    reqs, exp_map = mod.select_darp_requests_from_pool(rows, assignments)
    assert len(reqs) == 2
    assert exp_map[0] == 1 and exp_map[1] == 2
    assert reqs[0]["original_request_id"] == 0 and reqs[1]["original_request_id"] == 0

    assignments2 = [("no_MT", None)]
    reqs2, exp2 = mod.select_darp_requests_from_pool(rows, assignments2)
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
