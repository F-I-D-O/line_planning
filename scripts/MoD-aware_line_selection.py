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

import line_planning.instance
import line_planning.line_planning


def solution_to_darp_requests(
    line_instance: "line_planning.instance.line_instance",
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


line_planning_path = Path(r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning")

iteration_count = 2

test_data_path = Path(__file__).parent.parent / "test_data"
candidate_lines_file = test_data_path / "all_lines_nodes_100_c5.txt"
# Distance matrix file
dm_file = line_planning_path / "Instances/original/dm.h5"
demand_file = test_data_path / "OD_matrix_april_fhv_10_percent.txt"

results_dir_path = line_planning_path / "Results/original_instances/10_percent/budget_200000/mod-aware"

# Loads candidate lines file and similar: once per the whole script
line_inst = line_planning.instance.line_instance(
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

solver = line_planning.line_planning.line_planning_solver(line_inst)

instance_size_label = line_planning.line_planning.get_instance_size_label(demand_file)

for i in range(iteration_count):
    logging.info(f"Iteration {i+1} of {iteration_count}")

    results_dir_path_per_iteration = results_dir_path / f"iteration_{i+1}"
    results_dir_path_per_iteration.mkdir(parents=True)

    # 1. Solve the line selection ILP (section 4.1); get selected lines and request-line assignments
    obj_val, run_time_ILP, selected_lines, request_assignments = solver.solve_MoD_aware_ILP(
        export_model=True,
        export_solution=True,
        output_dir=results_dir_path_per_iteration,
        gurobi_log_file=results_dir_path_per_iteration / "gurobi.log",
    )

    # 2. Build MoD requests for DARP per section 4.2.1 (Conventional model) and write requests.csv
    darp_requests = solution_to_darp_requests(
        line_inst,
        request_assignments,
        request_times=None,
        delta_transfer_seconds=0,
        max_frequency=line_planning.line_planning.max_frequency,
    )
    write_darp_requests_csv(darp_requests, results_dir_path_per_iteration / "requests.csv", time_format="seconds")

    results_payload = {
            "objective_value": obj_val,
            "run_time_seconds": run_time_ILP,
            "instance_size": instance_size_label,
            "demand_file": str(demand_file),
            "n_darp_requests": len(darp_requests),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    results_file = results_dir_path_per_iteration / "metrics.json"
    results_file.write_text(json.dumps(results_payload, indent=2))
