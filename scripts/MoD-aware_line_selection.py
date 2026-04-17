"""
MoD-aware line selection script: solve section 4.1 ILP and produce DARP request files
per section 4.2.1 (Conventional model) for the Ridesharing_DARP_instances format.
"""

import csv
import logging
import os
from pathlib import Path
import sys
import time
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import yaml

import darpinstances.inout

import darpbenchmark.experiments

import lineplanning.instance
import lineplanning.line_planning


experiment_data_path = Path(r"C:\Google Drive AIC\My Drive\AIC Experiment Data")


iteration_count = 30
line_planning_ILP_time_limit = 1500 # seconds
rejection_cost = 1000 # in the travel time units, i.e. seconds of travel time

# Upper bound on per-route frequency y_ρ in the MoD-aware ILP (manuscript §4.1.1).
max_frequency = 20

line_planning_path = experiment_data_path / "Line Planning"
demand_file = None
dm_file = None

# instance_dir = experiment_data_path / "DARP/Instances/Manhattan"
# candidate_lines_file = instance_dir / "lines.txt"
# dm_file = instance_dir / "dm.h5"
# demand_file = instance_dir / "instances/start_18-00/duration_02_h/max_delay_05_min/requests.csv"
# results_dir_path = line_planning_path / "Results/manhattan_test/mod-aware"

# Chyse
# instance_dir = experiment_data_path / "Line Planning/Instances/Chyse"
# candidate_lines_file = instance_dir / "lines.txt"
# dm_file = instance_dir / "dm.csv"
# demand_file = instance_dir / "requests.csv"
# results_dir_path = line_planning_path / "Results/chyse_test/mod-aware"

# Manhattan 10% demand
instance_dir = experiment_data_path / "Line Planning/Instances/manhattan-2_h-10_percent/instance_01"
candidate_lines_file = instance_dir / "lines.txt"
results_dir_path = line_planning_path / "Results/manhattan-2_h-10_percent/instance_01/mod-aware"

def setup_file_logging(results_dir: Path) -> Path:
    """
    Configure root logging to DEBUG and write a log file in results_dir.
    Returns the log file path.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"mod_aware_line_selection_{time.strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if this script is re-run in an interactive session.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    logging.debug("File logging initialized")
    logging.info("Logging to %s", log_path)
    return log_path


def _resolve_config_path(instance_dir_path: Path) -> Path:
    config_candidates = [
        instance_dir_path / "config.yaml",
        instance_dir_path / "config.yml",
        instance_dir_path / "instance_config.yaml",
        instance_dir_path / "instance_config.yml",
    ]
    for p in config_candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find an instance config file to infer demand/dm paths. "
        f"Tried: {[str(p) for p in config_candidates]}"
    )


def _coerce_path(v: object) -> Path:
    if isinstance(v, Path):
        return v
    if isinstance(v, str):
        return Path(v)
    raise TypeError(f"Expected str/Path, got {type(v).__name__}")


def _load_demand_and_dm_from_instance_config(instance_dir_path: Path) -> Tuple[Path, Path]:
    config_path = _resolve_config_path(instance_dir_path)
    config_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a mapping at top-level in {config_path}, got {type(cfg).__name__}.")

    demand_cfg = cfg.get("demand")
    dm_filepath = cfg.get("dm_filepath")

    if not isinstance(demand_cfg, dict):
        raise ValueError(f"Expected config['demand'] to be a mapping in {config_path}.")

    demand_filepath = demand_cfg.get("filepath")

    if not demand_filepath:
        raise KeyError(
            f"Missing demand filepath in {config_path}. Expected keys like demand.filepath."
        )
    if not dm_filepath:
        raise KeyError(
            f"Missing dm filepath in {config_path}. Expected key like dm_filepath."
        )

    demand_path = _coerce_path(demand_filepath)
    dm_path = _coerce_path(dm_filepath)

    if not demand_path.is_absolute():
        demand_path = (config_dir / demand_path).resolve()
    if not dm_path.is_absolute():
        dm_path = (config_dir / dm_path).resolve()

    return demand_path, dm_path


if demand_file is None or dm_file is None:
    inferred_demand_file, inferred_dm_file = _load_demand_and_dm_from_instance_config(instance_dir)
    if demand_file is None:
        demand_file = inferred_demand_file
    if dm_file is None:
        dm_file = inferred_dm_file

if demand_file is None or dm_file is None:
    raise ValueError(
        "demand_file and dm_file must be set either directly in the script or via instance_dir/config.yaml."
    )

# Initialize debug file logging in the (top-level) results directory.
_log_path = setup_file_logging(results_dir_path)

# Perivier instance - broken triangle inequality
# test_data_path = Path(__file__).parent.parent / "test_data"
# candidate_lines_file = test_data_path / "all_lines_nodes_100_c5.txt"
# # Distance matrix file
# dm_file = line_planning_path / "Instances/original/dm.h5"
# demand_file = test_data_path / "OD_matrix_april_fhv_10_percent.txt"
# results_dir_path = line_planning_path / "Results/original_instances/10_percent/budget_200000/mod-aware"


DARP_BENCHMARK_PATH = Path(r"C:/Workspaces/AIC/DARP-Benchmark/cmake-build-release/DARP-benchmark")


def solution_to_darp_requests(
    line_instance: "lineplanning.instance.line_instance",
    request_assignments: List[Tuple[str, Optional[int]]],
    request_times: Optional[List[Union[int, float]]] = None,
    delta_transfer_seconds: Union[int, float] = 0,
) -> List[dict]:
    """
    Build the set of MoD requests R_MoD for the conventional model (section 4.2.1).

    For each original request r:
    - If assigned to MoD-only: one request (o_r, d_r, t_r).
    - If assigned to line ℓ: first-mile (o_r, s^b_ℓr, t_r) and last-mile (s^u_ℓr, d_r, t_unboard_r).
    - If rejected (ILP §4.1.2): omitted from R_MoD (no DARP rows).

    Boarding time is estimated as t_board = t_r + ftt(o_r, s^b_ℓr) + δ_transfer (using dm as travel time).
    Unboarding time: t_unboard = t_board + segment_travel_time on MT (approximated from line travel time).

    Returns a list of request dicts with keys: origin, destination, time, id (0-based DARP request index),
    original_request_id (0-based index of the original line-planning request).
    Format matches https://github.com/aicenter/Ridesharing_DARP_instances (requests.csv) plus original_request_id.
    """
    nb_pass = line_instance.nb_pass
    if request_times is None:
        request_times = [0] * nb_pass
    if len(request_times) != nb_pass:
        request_times = request_times + [0] * (nb_pass - len(request_times))

    dm = line_instance.dm
    requests_od = line_instance.requests
    set_of_lines = line_instance.set_of_lines
    lengths_travel_times = line_instance.lengths_travel_times

    darp_requests = []
    request_id = 0

    for r in range(nb_pass):
        o_r = requests_od[r][0]
        d_r = requests_od[r][1]
        t_r = float(request_times[r])

        kind, line_idx = request_assignments[r]
        if kind == "rejected":
            continue
        if kind == "no_MT" or line_idx is None:
            # Direct MoD trip: (o_r, d_r, t_r)
            darp_requests.append({
                "id": request_id,
                "original_request_id": r,
                "origin": o_r,
                "destination": d_r,
                "time": t_r,
            })
            request_id += 1
            continue

        # Assigned to route ρ (candidate line index); MoD-aware ILP uses route-aggregated variables (§4.1.1)
        route = line_idx
        opt = line_instance.trip_option_on_line(r, route)
        if opt is None:
            raise ValueError(f"Request {r} has no feasible trip data for route {route}")
        sb = opt.mt_pickup_node
        su = opt.mt_drop_off_node

        # First-mile request: (o_r, s^b_ℓr, t_r)
        darp_requests.append({
            "id": request_id,
            "original_request_id": r,
            "origin": o_r,
            "destination": sb,
            "time": t_r,
        })
        request_id += 1

        # t_board ≈ t_r + ftt(o_r, s^b) + δ_transfer (equation 10; using dm as travel time)
        first_mile_time = float(dm[o_r][sb]) if dm is not None and sb >= 0 else 0.0
        t_board = t_r + first_mile_time + delta_transfer_seconds

        # Segment travel time on line (boarding to unboarding)
        line_length = set_of_lines[route][0]
        if line_length > 0 and lengths_travel_times is not None:
            segment_edges = max(0, opt.mt_drop_off_line_edge_index - opt.mt_pickup_line_edge_index)
            segment_time = lengths_travel_times[route] * (segment_edges / line_length)
        else:
            segment_time = 0.0
        t_unboard = t_board + segment_time

        # Last-mile request: (s^u_ℓr, d_r, t_unboard_r)
        darp_requests.append({
            "id": request_id,
            "original_request_id": r,
            "origin": su,
            "destination": d_r,
            "time": t_unboard,
        })
        request_id += 1

    return darp_requests


def load_darp_requests_csv(path: Union[Path, str]) -> List[dict]:
    """
    Load DARP requests from a CSV file.

    Returns a list of dicts with keys: origin, destination, time, id, original_request_id.
    """
    path = Path(path)
    darp_requests = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            darp_requests.append({
                "id": int(row["id"]),
                "original_request_id": int(row["original_request_id"]),
                "origin": int(row["origin"]),
                "destination": int(row["destination"]),
                "time": float(row["time"]),
            })
    return darp_requests


def load_request_assignments_csv(
    path: Union[Path, str],
) -> List[Tuple[str, Optional[int]]]:
    """
    Load request assignments from passenger_assignments.csv (exported by the ILP solver).

    The CSV has columns: passenger, line, mod_cost
    where `line` is either:
    - an integer (route index ρ) if assigned to a line
    - "no_MT" if assigned to MoD-only
    - "rejected" if not served (§4.1.2)
    - "Dropped" if dropped (treated as no_MT here)

    Returns a list of (kind, route_index) tuples where kind is "no_MT", "line", or "rejected";
    for "line", the second entry is the route index ρ (0 .. nb_lines-1).
    """
    import pandas as pd
    path = Path(path)
    df = pd.read_csv(path)
    df = df.sort_values("passenger").reset_index(drop=True)

    request_assignments = []
    for _, row in df.iterrows():
        line_value = row["line"]
        if line_value == "no_MT":
            request_assignments.append(("no_MT", None))
        elif line_value == "rejected":
            request_assignments.append(("rejected", None))
        else:
            route_index = int(line_value)
            request_assignments.append(("line", route_index))
    return request_assignments


def write_darp_requests_csv(
    darp_requests: List[dict],
    path: Union[Path, str],
    time_format: str = "seconds",
) -> None:
    """
    Write DARP requests to a CSV file in Ridesharing_DARP_instances format.

    Columns: origin, destination, time, id, original_request_id.
    time_format: "seconds" (numeric) or "datetime" (yyyy-mm-dd HH:MM:SS).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["origin", "destination", "time", "id", "original_request_id"])
        w.writeheader()
        for req in darp_requests:
            row = {
                "origin": req["origin"],
                "destination": req["destination"],
                "time": req["time"] if time_format == "seconds" else req.get("time_datetime", req["time"]),
                "id": req["id"],
                "original_request_id": req["original_request_id"],
            }
            w.writerow(row)


def write_darp_vehicles_csv(
    darp_requests: List[dict],
    path: Union[Path, str],
    capacity: int = 5,
    time_format: str = "seconds",
) -> None:
    """
    Write DARP vehicles to a CSV file in Ridesharing_DARP_instances format.

    One vehicle is created per request, with:
    - starting position = request's pickup (origin) position
    - operation_start = request's pickup time
    - capacity = given capacity (default 5)

    Columns: id, origin, capacity, operation_start
    time_format: "seconds" (numeric) or "datetime" (yyyy-mm-dd HH:MM:SS).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "position", "capacity", "operation_start"])
        w.writeheader()
        for req in darp_requests:
            row = {
                "id": req["id"],
                "position": req["origin"],
                "capacity": capacity,
                "operation_start": req["time"] if time_format == "seconds" else req.get("time_datetime", req["time"]),
            }
            w.writerow(row)


def write_darp_config_yaml(
    output_dir: Union[Path, str],
    dm_filepath: Union[Path, str],
    max_travel_time_delay_seconds: int = 300,
    vehicle_capacity: int = 5,
) -> None:
    """
    Write DARP instance and experiment config YAML files for DARP-benchmark.

    Creates two files in output_dir:
    - config.yaml: Instance configuration (requests, vehicles, dm, constraints)
    - experiment_ih.yaml: Experiment configuration for running insertion heuristic

    See: https://github.com/aicenter/DARP-benchmark
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_config = {
        "demand": {
            "filepath": "./requests.csv",
        },
        "vehicles": {
            "filepath": "./vehicles.csv",
            "vehicle_capacity": vehicle_capacity,
        },
        "dm_filepath": str(dm_filepath),
        "max_travel_time_delay": {
            "mode": "absolute",
            "seconds": max_travel_time_delay_seconds,
        },
    }

    instance_config_path = output_dir / "config.yaml"
    with instance_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(instance_config, f, default_flow_style=False, sort_keys=False)

    experiment_config = {
        "instance": "./config.yaml",
        "method": "ih",
        "outdir": ".",
    }

    experiment_config_path = output_dir / "experiment_ih.yaml"
    with experiment_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(experiment_config, f, default_flow_style=False, sort_keys=False)


def _darp_id_to_line_planning_leg(
    darp_requests: List[dict],
    request_assignments: List[Tuple[str, Optional[int]]],
) -> Dict[int, str]:
    """
    Map each DARP request id to how it relates to line-planning MoD legs:

    - ``no_mt``: direct MoD (one DARP row per original request).
    - ``first_mile``: first DARP row for an MT passenger (o_r → s^b), from ``solution_to_darp_requests``.
    - ``last_mile``: second DARP row (s^u → d_r).

    Row order follows construction in ``solution_to_darp_requests`` (lower id = first mile).
    """
    by_original: Dict[int, List[int]] = {}
    for row in darp_requests:
        oid = int(row["original_request_id"])
        by_original.setdefault(oid, []).append(int(row["id"]))

    out: Dict[int, str] = {}
    for oid, ids in by_original.items():
        ids_sorted = sorted(ids)
        kind, line_idx = request_assignments[oid]
        assert kind != "rejected", f"DARP row(s) for original {oid} but assignment is rejected"
        if kind == "no_MT" or line_idx is None:
            assert len(ids_sorted) == 1, (
                f"original {oid} no_MT: expected 1 DARP id, got {ids_sorted}"
            )
            out[ids_sorted[0]] = "no_mt"
        else:
            assert len(ids_sorted) == 2, (
                f"original {oid} line assignment: expected 2 DARP ids, got {ids_sorted}"
            )
            out[ids_sorted[0]] = "first_mile"
            out[ids_sorted[1]] = "last_mile"
    return out


def compute_per_darp_request_costs(
    solution: dict,
    darp_requests: List[dict],
    request_assignments: List[Tuple[str, Optional[int]]],
) -> Dict[int, Tuple[float, float]]:
    """
    DARP plan cost shares for each MoD request, aligned with **line-planning** first/last mile.

    Each DARP row is one MoD trip (pickup + drop_off) in **some** vehicle plan; that plan's
    ``cost`` is split evenly across all requests served in that plan
    (``plan_cost / num_requests``). That scalar is the full MoD cost share for that row.

    For passengers using MT, the ILP produces **two** DARP rows (first-mile leg and last-mile
    leg); they may appear in **different** plans. This function uses ``request_assignments``
    and row order (see ``_darp_id_to_line_planning_leg``) to map each row's share into
    ``(line_plan_first_mile_component, line_plan_last_mile_component)``:

    - first-mile row: ``(share, 0)``
    - last-mile row: ``(0, share)``
    - direct MoD row: ``(share, 0)`` (entire trip attributed to the direct leg field in aggregation)

    Args:
        solution: DARP solution dict with ``plans`` list.
        darp_requests: All DARP rows for this run (ids must match solution ``request_index``).
        request_assignments: Same ILP output used to build ``requests.csv``.

    Returns:
        Dict mapping DARP id -> ``(first_mile_slot, last_mile_slot)`` cost for updates.
    """
    plan_share: Dict[int, float] = {}

    for plan in solution["plans"]:
        actions = plan["actions"]
        if not actions:
            continue

        plan_cost = float(plan["cost"])
        assert len(actions) % 2 == 0, "each DARP request in a plan must have pickup and drop_off"
        num_requests = len(actions) // 2
        assert num_requests > 0

        cost_per_request = plan_cost / num_requests

        pickups: Dict[int, None] = {}
        dropoffs: Dict[int, None] = {}
        for action in actions:
            a = action["action"]
            rid = int(a["request_index"])
            typ = a["type"]
            if typ == "pickup":
                assert rid not in pickups, f"DARP request {rid}: duplicate pickup in plan"
                pickups[rid] = None
            elif typ == "drop_off":
                assert rid not in dropoffs, f"DARP request {rid}: duplicate drop_off in plan"
                dropoffs[rid] = None
            else:
                raise AssertionError(f"unexpected action type {typ!r} for request {rid}")

        assert pickups.keys() == dropoffs.keys(), (
            f"pickup/drop_off mismatch: pickups={sorted(pickups)} dropoffs={sorted(dropoffs)}"
        )
        assert len(pickups) == num_requests, (
            f"expected {num_requests} requests in plan, got {len(pickups)} distinct indices"
        )

        for rid in pickups:
            assert rid not in plan_share, f"DARP request {rid} appears in more than one plan"
            plan_share[rid] = cost_per_request

    expected = {int(r["id"]) for r in darp_requests}
    assert plan_share.keys() == expected, (
        f"DARP cost extraction: ids in solution {sorted(plan_share.keys())} != "
        f"ids in requests.csv {sorted(expected)}"
    )

    leg_kind = _darp_id_to_line_planning_leg(darp_requests, request_assignments)
    assert leg_kind.keys() == expected, "leg map must cover every DARP id"

    result: Dict[int, Tuple[float, float]] = {}
    for rid, share in plan_share.items():
        kind = leg_kind[rid]
        if kind == "first_mile":
            result[rid] = (share, 0.0)
        elif kind == "last_mile":
            result[rid] = (0.0, share)
        else:
            assert kind == "no_mt"
            result[rid] = (share, 0.0)

    return result


def aggregate_mod_costs_for_original_requests(
    darp_request_leg_costs: Dict[int, Tuple[float, float]],
    darp_requests: List[dict],
    request_assignments: List[Tuple[str, Optional[int]]],
) -> dict:
    """
    Aggregate DARP per-request (pickup, dropoff) cost shares back to original line-planning
    requests (section 4.3.2).

    For MoD-only original requests: ``direct_trip_options`` gets the pickup and dropoff
    shares from the single DARP request.

    For line-assigned original requests: first-mile DARP row's pickup+dropoff sum updates
    ``first_mile_cost``; last-mile DARP row's sum updates ``last_mile_cost``.

    Args:
        darp_request_leg_costs: DARP request id -> (pickup_leg_cost, dropoff_leg_cost).
        darp_requests: List of DARP request dicts with 'id' and 'original_request_id'.
        request_assignments: List of (kind, line_idx) tuples for original requests.

    Returns:
        Dict mapping original_request_id -> (first_mile_cost, last_mile_cost).
    """
    original_request_costs: dict = {}

    for original_id in range(len(request_assignments)):
        kind, line_idx = request_assignments[original_id]
        if kind == "rejected":
            continue

        darp_ids_for_original = sorted(
            int(req["id"]) for req in darp_requests if req["original_request_id"] == original_id
        )

        if kind == "no_MT" or line_idx is None:
            assert len(darp_ids_for_original) == 1, (
                f"MoD-only original request {original_id} expected 1 DARP request, "
                f"found {len(darp_ids_for_original)}"
            )
            darp_id = darp_ids_for_original[0]
            fm_dl, lm_dl = darp_request_leg_costs[darp_id]
            original_request_costs[original_id] = (float(fm_dl), float(lm_dl))
        else:
            assert len(darp_ids_for_original) == 2, (
                f"Line-assigned original request {original_id} expected 2 DARP requests, "
                f"found {len(darp_ids_for_original)}"
            )
            first_mile_darp_id = darp_ids_for_original[0]
            last_mile_darp_id = darp_ids_for_original[1]
            f1, l1 = darp_request_leg_costs[first_mile_darp_id]
            f2, l2 = darp_request_leg_costs[last_mile_darp_id]
            original_request_costs[original_id] = (
                float(f1) + float(l1),
                float(f2) + float(l2),
            )

    return original_request_costs


def _get_current_costs_for_assignments(
    line_instance: "lineplanning.instance.line_instance",
    request_assignments: List[Tuple[str, Optional[int]]],
) -> Dict[int, Tuple[float, float]]:
    """
    Extract the currently stored MoD (first_mile_cost, last_mile_cost) for each *served*
    original request, keyed by original_request_id, based on the current request assignment.

    This is used to compare "old" ILP costs vs recomputed costs from a DARP run.
    """
    costs: Dict[int, Tuple[float, float]] = {}
    for original_id, (kind, line_idx) in enumerate(request_assignments):
        if kind == "rejected":
            continue
        if kind == "no_MT" or line_idx is None:
            opt = line_instance.direct_trip_options[original_id]
        else:
            opt = line_instance.optimal_trip_options[original_id][int(line_idx)]
        costs[original_id] = (float(opt.first_mile_cost), float(opt.last_mile_cost))
    return costs


def _avg_cost_per_traveltime_across_requests(
    costs: Dict[int, Tuple[float, float]],
    line_instance: "lineplanning.instance.line_instance",
) -> float:
    """
    Compute average over requests of (total_cost / direct_travel_time).

    direct_travel_time uses the instance dm between original OD nodes. Requests with
    non-positive travel time are skipped.
    """
    dm = line_instance.dm
    reqs = line_instance.requests

    total = 0.0
    n = 0
    for original_id, (c1, c2) in costs.items():
        o, d = reqs[original_id][0], reqs[original_id][1]
        tt = float(dm[o][d]) if dm is not None else 0.0
        if tt <= 0.0:
            continue
        total_cost = float(c1) + float(c2)
        total += (total_cost / tt)
        n += 1

    return (total / n) if n > 0 else 0.0


def _scale_costs(
    costs: Dict[int, Tuple[float, float]],
    factor: float,
) -> Dict[int, Tuple[float, float]]:
    if factor == 1.0:
        return dict(costs)
    return {k: (v[0] * factor, v[1] * factor) for k, v in costs.items()}


def _scale_all_mod_costs_in_model(
    line_instance: "lineplanning.instance.line_instance",
    factor: float,
) -> None:
    """
    Multiply *all* MoD costs stored in the model by `factor` (not only those used in the last solution):
    - direct_trip_options (no_MT option per request)
    - optimal_trip_options (all per-request per-route options)
    """
    if factor == 1.0:
        return

    # Scale direct trips (no_MT) for every request
    for p, opt in enumerate(line_instance.direct_trip_options):
        line_instance.direct_trip_options[p] = opt._replace(
            first_mile_cost=float(opt.first_mile_cost) * factor,
            last_mile_cost=float(opt.last_mile_cost) * factor,
        )

    # Scale all line-based options for every request / route
    for p, opts_by_route in enumerate(line_instance.optimal_trip_options):
        # opts_by_route is a dict: route_index -> TripOption
        for rho, opt in list(opts_by_route.items()):
            opts_by_route[rho] = opt._replace(
                first_mile_cost=float(opt.first_mile_cost) * factor,
                last_mile_cost=float(opt.last_mile_cost) * factor,
            )


@dataclass(frozen=True)
class ModCostRecomputeContext:
    """
    Shared inputs for MoD cost recomputation strategies.
    """

    line_instance: "lineplanning.instance.line_instance"
    request_assignments: List[Tuple[str, Optional[int]]]
    darp_request_leg_costs: Dict[int, Tuple[float, float]]
    darp_requests: List[dict]


ModCostRecomputeStrategy = Callable[[ModCostRecomputeContext], Dict[int, Tuple[float, float]]]


def recompute_costs__aggregate_original_requests(
    ctx: ModCostRecomputeContext,
) -> Dict[int, Tuple[float, float]]:
    """
    Baseline: aggregate DARP per-request costs back to original requests, then
    ``solver.update_mod_costs`` writes them into ``optimal_trip_options`` and
    ``direct_trip_options`` (pure MoD). The MoD-aware ILP reads those stored
    values for both line legs and ``no_MT``.
    """
    return aggregate_mod_costs_for_original_requests(
        ctx.darp_request_leg_costs,
        ctx.darp_requests,
        ctx.request_assignments,
    )


def recompute_costs__rescale_avg_cost_per_traveltime_and_request(
    ctx: ModCostRecomputeContext,
) -> Dict[int, Tuple[float, float]]:
    """
    Alternative strategy:
    - Aggregate DARP costs back to original requests
    - Compute mean of (cost / direct_tt) for:
        - old costs (line-planning estimates before this update)
        - new costs (DARP aggregation on the served pattern)
    - Global factor ``factor = new_avg / old_avg`` pushes all MoD coefficients toward
      DARP's average cost-per-travel-time (not preserving the previous scale).
    - Apply ``factor`` to every MoD cost in the instance; ``update_mod_costs`` then
      overwrites the chosen option with ``factor`` times the DARP aggregate.
    """
    old_costs = _get_current_costs_for_assignments(ctx.line_instance, ctx.request_assignments)
    new_costs = recompute_costs__aggregate_original_requests(ctx)

    old_avg = _avg_cost_per_traveltime_across_requests(old_costs, ctx.line_instance)
    new_avg = _avg_cost_per_traveltime_across_requests(new_costs, ctx.line_instance)

    if new_avg <= 0.0:
        logging.warning(
            "MoD cost rescaling skipped because new_avg<=0 (old_avg=%s, new_avg=%s).",
            old_avg,
            new_avg,
        )
        return new_costs

    if old_avg <= 0.0:
        logging.warning(
            "MoD cost rescaling skipped because old_avg<=0 (old_avg=%s, new_avg=%s).",
            old_avg,
            new_avg,
        )
        return new_costs

    factor = new_avg / old_avg
    logging.info(
        "MoD cost rescaling factor (new_avg/old_avg, push toward DARP): old_avg=%s new_avg=%s factor=%s",
        old_avg,
        new_avg,
        factor,
    )

    # Apply the ratio to *all* MoD costs in the model (not only those used in the last ILP solution).
    _scale_all_mod_costs_in_model(ctx.line_instance, factor)

    # DARP aggregates scaled by the same factor; update_mod_costs overwrites chosen options.
    return _scale_costs(new_costs, factor)


MOD_COST_RECOMPUTE_STRATEGIES: Dict[str, ModCostRecomputeStrategy] = {
    # Baseline / reference implementation (current script behavior)
    "aggregate_original_requests": recompute_costs__aggregate_original_requests,
    # First alternative strategy (requested)
    "rescale_avg_cost_per_traveltime_and_request": (
        recompute_costs__rescale_avg_cost_per_traveltime_and_request
    ),
}

# Configure which strategy to use for MoD cost recomputation.
MOD_COST_RECOMPUTE_STRATEGY = "rescale_avg_cost_per_traveltime_and_request"



# Loads candidate lines file and similar: once per the whole script
line_inst = lineplanning.instance.line_instance(
    candidate_lines_file=candidate_lines_file,
    capacity=30,
    maximum_detour=3,
    demand_file=demand_file,
    preprocessing_dir=instance_dir / "preprocessing",
    dm_file=dm_file,
)

solver = lineplanning.line_planning.LinePlanningSolver(
    line_inst,
    time_limit=line_planning_ILP_time_limit,
    cost_coefficient=1.0,
    max_frequency=max_frequency
)

instance_size_label = lineplanning.line_planning.get_instance_size_label(demand_file)

try:
    for i in range(iteration_count):
        logging.info("Iteration %s of %s", i + 1, iteration_count)

        results_dir_path_per_iteration = results_dir_path / f"iteration_{i+1}"
        results_dir_path_per_iteration.mkdir(parents=True, exist_ok=True)

        # Check if DARP input files already exist; if so, skip to step 2.2
        requests_csv_path = results_dir_path_per_iteration / "requests.csv"
        passenger_assignments_csv_path = results_dir_path_per_iteration / "passenger_assignments.csv"
        experiment_config_path = results_dir_path_per_iteration / "experiment_ih.yaml"

        if requests_csv_path.exists() and passenger_assignments_csv_path.exists() and experiment_config_path.exists():
            logging.info(
                "Iteration %s: DARP input files exist, skipping ILP and loading from files",
                i + 1,
            )
            darp_requests = load_darp_requests_csv(requests_csv_path)
            request_assignments = load_request_assignments_csv(
                passenger_assignments_csv_path,
            )
        else:
            # 1. Solve the line selection ILP (section 4.1); get selected lines and request-line assignments
            # Same formulation as experiment ``solver.method: non_budget_ilp`` in ``run_experiment``.
            obj_val, run_time_ILP, selected_lines, request_assignments, line_cost, mod_cost = (
                solver.solve_MoD_aware_ILP(
                    export_model=True,
                    export_solution=True,
                    output_dir=results_dir_path_per_iteration,
                    gurobi_log_file=results_dir_path_per_iteration / "gurobi.log",
                    max_route_frequency=max_frequency,
                    rejection_cost=rejection_cost,
                    use_request_line_valid_inequalities=True,
                    reuse_model=True,
                )
            )
            # 1.2 Write metrics.json
            results_payload = {
                "iteration": i + 1,
                "objective_value": obj_val,
                "line_cost": line_cost,
                "mod_cost": mod_cost,
                "run_time_seconds": run_time_ILP,
                "instance_size": instance_size_label,
                "demand_file": str(demand_file),
                "max_route_frequency": max_frequency,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            results_file = results_dir_path_per_iteration / "metrics.json"
            results_file.write_text(json.dumps(results_payload, indent=2))

            # 2. DARP
            # Build MoD requests for DARP per section 4.2.1 (Conventional model) and write requests.csv
            darp_requests = solution_to_darp_requests(
                line_inst,
                request_assignments,
                request_times=None,
                delta_transfer_seconds=0,
            )
            write_darp_requests_csv(darp_requests, requests_csv_path, time_format="seconds")
            write_darp_vehicles_csv(
                darp_requests,
                results_dir_path_per_iteration / "vehicles.csv",
                capacity=5,
                time_format="seconds",
            )
            write_darp_config_yaml(
                output_dir=results_dir_path_per_iteration,
                dm_filepath=dm_file,
                max_travel_time_delay_seconds=300,
                vehicle_capacity=5,
            )

        # 2.2 call DARP solver
        darpbenchmark.experiments.run_experiment_using_config(
            experiment_config_path,
            executable_path=DARP_BENCHMARK_PATH,
        )

        # 3. Recompute the MoD cost estimates (section 4.3.2)
        solution_path = results_dir_path_per_iteration / "config.yaml-solution.json"
        darp_solution = darpinstances.inout.load_json(solution_path)

        darp_request_leg_costs = compute_per_darp_request_costs(
            darp_solution,
            darp_requests,
            request_assignments,
        )

        strategy = MOD_COST_RECOMPUTE_STRATEGIES.get(MOD_COST_RECOMPUTE_STRATEGY)
        if strategy is None:
            raise KeyError(
                f"Unknown MOD_COST_RECOMPUTE_STRATEGY={MOD_COST_RECOMPUTE_STRATEGY!r}. "
                f"Available: {sorted(MOD_COST_RECOMPUTE_STRATEGIES.keys())}"
            )
        ctx = ModCostRecomputeContext(
            line_instance=line_inst,
            request_assignments=request_assignments,
            darp_request_leg_costs=darp_request_leg_costs,
            darp_requests=darp_requests,
        )
        new_mod_costs = strategy(ctx)

        solver.update_mod_costs(new_mod_costs, request_assignments)
        logging.info("Iteration %s: updated MoD costs for %s requests", i + 1, len(new_mod_costs))
except Exception:
    logging.exception("Uncaught error in MoD-aware line selection script")
    raise

