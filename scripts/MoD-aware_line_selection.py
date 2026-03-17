"""
MoD-aware line selection script: solve section 4.1 ILP and produce DARP request files
per section 4.2.1 (Conventional model) for the Ridesharing_DARP_instances format.
"""

import csv
import logging
from pathlib import Path
import time
import json
from typing import List, Optional, Tuple, Union

import yaml

import darpinstances.inout

import darpbenchmark.experiments

import lineplanning.instance
import lineplanning.line_planning


experiment_data_path = Path(r"C:\Google Drive AIC\My Drive\AIC Experiment Data")


iteration_count = 2

instance_dir = experiment_data_path / "DARP/Instances/Manhattan"
candidate_lines_file = instance_dir / "lines.txt"
dm_file = instance_dir / "dm.h5"
demand_file = instance_dir / "instances/start_18-00/duration_02_h/max_delay_05_min/requests.csv"
line_planning_path = experiment_data_path / "Line Planning"
results_dir_path = line_planning_path / "Results/manhattan_test/mod-aware"

# Perivier instance - broken triangle inequality
# test_data_path = Path(__file__).parent.parent / "test_data"
# candidate_lines_file = test_data_path / "all_lines_nodes_100_c5.txt"
# # Distance matrix file
# dm_file = line_planning_path / "Instances/original/dm.h5"
# demand_file = test_data_path / "OD_matrix_april_fhv_10_percent.txt"
# results_dir_path = line_planning_path / "Results/original_instances/10_percent/budget_200000/mod-aware"


DARP_BENCHMARK_PATH = Path(r"C:/Workspaces/AIC/DARP_Benchmark/cmake-build-debug/DARP-benchmark")


def solution_to_darp_requests(
    line_instance: "lineplanning.instance.line_instance",
    request_assignments: List[Tuple[str, Optional[int]]],
    request_times: Optional[List[Union[int, float]]] = None,
    delta_transfer_seconds: Union[int, float] = 0,
    max_frequency: int = 1,
) -> List[dict]:
    """
    Build the set of MoD requests R_MoD for the conventional model (section 4.2.1).

    For each original request r:
    - If assigned to MoD-only: one request (o_r, d_r, t_r).
    - If assigned to line ℓ: first-mile (o_r, s^b_ℓr, t_r) and last-mile (s^u_ℓr, d_r, t_unboard_r).

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

        # Assigned to line ℓ: first-mile and last-mile
        route = line_idx // max_frequency
        opt = line_instance.optimal_trip_options[r][route]
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
    max_frequency: int = 1,
) -> List[Tuple[str, Optional[int]]]:
    """
    Load request assignments from passenger_assignments.csv (exported by the ILP solver).

    The CSV has columns: passenger, line, mod_cost
    where `line` is either:
    - an integer (route_index) if assigned to a line
    - "no_MT" if assigned to MoD-only
    - "Dropped" if dropped (treated as no_MT here)

    Returns a list of (kind, line_idx) tuples where kind is "no_MT" or "line".
    line_idx is the line index (route_index * max_frequency), not the route_index.
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
        else:
            route_index = int(line_value)
            line_idx = route_index * max_frequency
            request_assignments.append(("line", line_idx))
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
        w = csv.DictWriter(f, fieldnames=["id", "origin", "capacity", "operation_start"])
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


def compute_per_darp_request_costs(solution: dict) -> dict:
    """
    Compute the cost share for each DARP request from the DARP solution (section 4.3.2).

    For each plan, the cost per request is: plan_cost / num_requests_in_plan.
    Each request contributes equally to the total plan cost.

    Args:
        solution: DARP solution dict with 'plans' key.

    Returns:
        Dict mapping DARP request_id -> cost share.
    """
    darp_request_costs = {}

    for plan in solution["plans"]:
        if len(plan["actions"]) == 0:
            continue

        plan_cost = plan["cost"]
        num_requests = len(plan["actions"]) // 2
        if num_requests == 0:
            continue

        cost_per_request = plan_cost / num_requests

        for action in plan["actions"]:
            request_id = action["action"]["request_index"]
            if request_id not in darp_request_costs:
                darp_request_costs[request_id] = cost_per_request

    return darp_request_costs


def aggregate_mod_costs_for_original_requests(
    darp_request_costs: dict,
    darp_requests: List[dict],
    request_assignments: List[Tuple[str, Optional[int]]],
) -> dict:
    """
    Aggregate DARP request costs back to original line-planning requests (section 4.3.2).

    For MoD-only original requests: cost = cost of the single direct trip DARP request.
    For line-assigned original requests: first_mile_cost + last_mile_cost from two DARP requests.

    Args:
        darp_request_costs: Dict mapping DARP request_id -> cost share.
        darp_requests: List of DARP request dicts with 'id' and 'original_request_id'.
        request_assignments: List of (kind, line_idx) tuples for original requests.

    Returns:
        Dict mapping original_request_id -> (first_mile_cost, last_mile_cost).
    """
    original_request_costs: dict = {}

    for original_id in range(len(request_assignments)):
        kind, line_idx = request_assignments[original_id]

        darp_ids_for_original = [
            req["id"] for req in darp_requests if req["original_request_id"] == original_id
        ]

        if kind == "no_MT" or line_idx is None:
            if len(darp_ids_for_original) != 1:
                raise ValueError(
                    f"MoD-only original request {original_id} expected 1 DARP request, "
                    f"found {len(darp_ids_for_original)}"
                )
            darp_id = darp_ids_for_original[0]
            cost = darp_request_costs.get(darp_id, 0.0)
            original_request_costs[original_id] = (cost, 0.0)
        else:
            if len(darp_ids_for_original) != 2:
                raise ValueError(
                    f"Line-assigned original request {original_id} expected 2 DARP requests, "
                    f"found {len(darp_ids_for_original)}"
                )
            first_mile_darp_id = darp_ids_for_original[0]
            last_mile_darp_id = darp_ids_for_original[1]
            first_mile_cost = darp_request_costs.get(first_mile_darp_id, 0.0)
            last_mile_cost = darp_request_costs.get(last_mile_darp_id, 0.0)
            original_request_costs[original_id] = (first_mile_cost, last_mile_cost)

    return original_request_costs




# Loads candidate lines file and similar: once per the whole script
line_inst = lineplanning.instance.line_instance(
    candidate_lines_file=candidate_lines_file,
    cost=1,
    max_length=15,
    min_length=8,
    proba=0.1,
    capacity=30,
    detour_factor=3,
    method=3,
    granularity=1,
    demand_file=demand_file,
    results_dir=results_dir_path,
    dm_file=dm_file,
)

solver = lineplanning.line_planning.LinePlanningSolver(line_inst)

instance_size_label = lineplanning.line_planning.get_instance_size_label(demand_file)

for i in range(iteration_count):
    logging.info(f"Iteration {i+1} of {iteration_count}")

    results_dir_path_per_iteration = results_dir_path / f"iteration_{i+1}"
    results_dir_path_per_iteration.mkdir(parents=True, exist_ok=True)

    # Check if DARP input files already exist; if so, skip to step 2.2
    requests_csv_path = results_dir_path_per_iteration / "requests.csv"
    passenger_assignments_csv_path = results_dir_path_per_iteration / "passenger_assignments.csv"
    experiment_config_path = results_dir_path_per_iteration / "experiment_ih.yaml"

    if requests_csv_path.exists() and passenger_assignments_csv_path.exists() and experiment_config_path.exists():
        logging.info(f"Iteration {i+1}: DARP input files exist, skipping ILP and loading from files")
        darp_requests = load_darp_requests_csv(requests_csv_path)
        request_assignments = load_request_assignments_csv(
            passenger_assignments_csv_path,
            max_frequency=lineplanning.line_planning.max_frequency,
        )
    else:
        # 1. Solve the line selection ILP (section 4.1); get selected lines and request-line assignments
        obj_val, run_time_ILP, selected_lines, request_assignments = solver.solve_MoD_aware_ILP(
            export_model=True,
            export_solution=True,
            output_dir=results_dir_path_per_iteration,
            gurobi_log_file=results_dir_path_per_iteration / "gurobi.log",
        )
        # 1.2 Write metrics.json
        results_payload = {
            "iteration": i + 1,
            "objective_value": obj_val,
            "run_time_seconds": run_time_ILP,
            "instance_size": instance_size_label,
            "demand_file": str(demand_file),
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
            max_frequency=lineplanning.line_planning.max_frequency,
        )
        write_darp_requests_csv(darp_requests, requests_csv_path, time_format="seconds")
        write_darp_vehicles_csv(darp_requests, results_dir_path_per_iteration / "vehicles.csv", capacity=5, time_format="seconds")
        write_darp_config_yaml(
            output_dir=results_dir_path_per_iteration,
            dm_filepath=dm_file,
            max_travel_time_delay_seconds=300,
            vehicle_capacity=5,
        )

    # 2.2 call DARP solver
    darpbenchmark.experiments.run_experiment_using_config(experiment_config_path, executable_path=DARP_BENCHMARK_PATH)

    # 3. Recompute the MoD cost estimates (section 4.3.2)
    solution_path = results_dir_path_per_iteration / "experiment_ih.yaml-solution.json"
    darp_solution = darpinstances.inout.load_json(solution_path)

    darp_request_costs = compute_per_darp_request_costs(darp_solution)
    new_mod_costs = aggregate_mod_costs_for_original_requests(
        darp_request_costs,
        darp_requests,
        request_assignments,
    )

    solver.update_mod_costs(new_mod_costs, request_assignments)
    logging.info(f"Iteration {i+1}: updated MoD costs for {len(new_mod_costs)} requests")

