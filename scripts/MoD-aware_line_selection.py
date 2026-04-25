"""
MoD-aware line selection script: solve section 4.1 ILP and produce DARP request files
per section 4.2.1 (Conventional model) for the Ridesharing_DARP_instances format.

Run::

    python MoD-aware_line_selection.py /path/to/experiment.yaml

Configuration is read entirely from the experiment YAML (same layout as ``lineplanning.line_planning.run_experiment``
where applicable: ``instance``, ``results_dir``, ``solver``, ``mass_transport``), plus MoD-specific keys.

Example ``experiment.yaml``::

    instance: ../../../Instances/my_case/instance_01/config.yaml
    results_dir: .
    initial_mod_cost_scale: 1.0
    iterations: 30

    solver:
      time_limit: 1500
      rejection_cost: 1000
      use_request_line_valid_inequalities: true
      reuse_model: true

    mass_transport:
      cost_coefficient: 1.0
      max_frequency: 20
      capacity: 30
      maximum_detour: 3
      line_mod_aggregate_prune: false

    darp:
      benchmark_executable: C:/path/to/DARP-benchmark  # optional; default: DARP_BENCHMARK_PATH / env
      transfer_delay: 0

    vehicles:
      vehicle_capacity: 5

    max_travel_time_delay:
      mode: absolute
      seconds: 300

    mod_cost_recomputation:
      strategy: aggregate_original_requests
      # For clustered_avg also set clusters: <int> and features: [travel_time, ...]
      # Optional: random_state, n_init (KMeans)
      smoothing:
        strategy: none
        under_relaxation_alpha: 0.3
"""

import argparse
import csv
import logging
import math
import os
from pathlib import Path
import sys
import time
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from sklearn.cluster import KMeans

import darpinstances.inout

import darpbenchmark.experiments

import lineplanning.instance
import lineplanning.instance_config
from lineplanning.instance_config import LinePlanningInstancePaths
import lineplanning.line_planning


# Defaults for code that imports this module without running ``main()`` (e.g. test scripts).
line_planning_ILP_time_limit = 1500  # seconds
rejection_cost = 1000  # in the travel time units, i.e. seconds of travel time
max_frequency = 20  # Upper bound on per-route frequency y_ρ in the MoD-aware ILP (manuscript §4.1.1).
DARP_BENCHMARK_PATH = Path(
    os.environ.get(
        "DARP_BENCHMARK_EXECUTABLE",
        r"C:/Workspaces/AIC/DARP-Benchmark/cmake-build-release/DARP-benchmark",
    )
)


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


def _coerce_path(v: object) -> Path:
    if isinstance(v, Path):
        return v
    if isinstance(v, str):
        return Path(v)
    raise TypeError(f"Expected str/Path, got {type(v).__name__}")


def _parse_initial_mod_cost_scale(cfg: dict, config_path: Path) -> float:
    """
    Optional ``initial_mod_cost_scale`` in experiment YAML (default 1.0).
    Multiplies every stored MoD leg cost (direct and line+MoD options) after loading the instance.
    """
    raw = cfg.get("initial_mod_cost_scale", 1.0)
    if raw is None:
        return 1.0
    try:
        s = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid initial_mod_cost_scale in {config_path!s}: expected a number, got {raw!r}."
        ) from exc
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError(
            f"Invalid initial_mod_cost_scale in {config_path!s}: expected a finite number > 0, got {s!r}."
        )
    return s


def _load_line_planning_paths_from_instance_dir(
    config_path: Path,
) -> Tuple[Path, Path, Path]:
    """
    Read the instance folder's config once and resolve demand/dm paths.
    ``config_path`` may be the instance ``config.yaml`` file or its parent directory (uses ``config.yaml``).
    Returns (demand_path, dm_path, instance_config_path).
    """
    config_path = Path(config_path).resolve()
    if config_path.is_dir():
        cand = config_path / "config.yaml"
        if not cand.is_file():
            raise FileNotFoundError(
                f"Instance path is a directory but no config file found at {cand}."
            )
        config_path = cand

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

    return demand_path, dm_path, config_path


EXPERIMENT_CONFIG_FILENAME = "experiment.yaml"

DARP_POOL_EXPORT_MAP_FILENAME = "darp_pool_export_map.json"


@dataclass(frozen=True)
class DarpPoolRow:
    """One canonical MoD leg row in the full DARP request pool (all trip options)."""

    pool_id: int
    passenger_idx: int
    route_idx: Optional[int]
    leg_kind: str
    origin: int
    destination: int
    time: float


@dataclass(frozen=True)
class ClusteredAvgState:
    """Extra inputs for ``clustered_avg`` recomputation (built each iteration after export)."""

    pool_rows: Tuple[DarpPoolRow, ...]
    pool_id_to_cluster: Dict[int, int]
    exported_id_to_pool_id: Dict[int, int]
    feature_names: Tuple[str, ...]
    n_clusters_requested: int


def _pool_feature_travel_time(
    line_instance: "lineplanning.instance.line_instance",
    row: DarpPoolRow,
) -> float:
    dm = line_instance.dm
    return float(dm[row.origin][row.destination])


POOL_FEATURE_BUILDERS: Dict[str, Callable[["lineplanning.instance.line_instance", DarpPoolRow], float]] = {
    "travel_time": _pool_feature_travel_time,
}


def _pool_rows_feature_matrix(
    line_instance: "lineplanning.instance.line_instance",
    pool_rows: Sequence[DarpPoolRow],
    feature_names: Tuple[str, ...],
) -> np.ndarray:
    for name in feature_names:
        if name not in POOL_FEATURE_BUILDERS:
            raise KeyError(
                f"Unknown mod_cost_recomputation feature {name!r}. "
                f"Available: {sorted(POOL_FEATURE_BUILDERS.keys())}"
            )
    X = np.zeros((len(pool_rows), len(feature_names)), dtype=float)
    for i, row in enumerate(pool_rows):
        for j, name in enumerate(feature_names):
            X[i, j] = float(POOL_FEATURE_BUILDERS[name](line_instance, row))
    return X


def build_darp_request_pool(
    line_instance: "lineplanning.instance.line_instance",
    request_times: Optional[List[Union[int, float]]] = None,
    transfer_delay: Union[int, float] = 0,
) -> List[DarpPoolRow]:
    """
    Full pool of DARP legs for every passenger direct trip and every feasible line option.

    Row timing and geometry match :func:`solution_to_darp_requests` for the same inputs.
    """
    nb_pass = line_instance.nb_pass
    if request_times is None:
        request_times = [0] * nb_pass
    if len(request_times) != nb_pass:
        request_times = list(request_times) + [0] * (nb_pass - len(request_times))

    dm = line_instance.dm
    requests_od = line_instance.requests
    set_of_lines = line_instance.set_of_lines
    lengths_travel_times = line_instance.lengths_travel_times

    rows: List[DarpPoolRow] = []
    pool_id = 0
    td = float(transfer_delay)

    for r in range(nb_pass):
        o_r = int(requests_od[r][0])
        d_r = int(requests_od[r][1])
        t_r = float(request_times[r])

        rows.append(
            DarpPoolRow(
                pool_id=pool_id,
                passenger_idx=r,
                route_idx=None,
                leg_kind="no_mt",
                origin=o_r,
                destination=d_r,
                time=t_r,
            )
        )
        pool_id += 1

        opts = line_instance.optimal_trip_options[r]
        for rho in sorted(opts.keys()):
            opt = line_instance.trip_option_on_line(r, rho)
            if opt is None:
                continue
            sb = int(opt.mt_pickup_node)
            su = int(opt.mt_drop_off_node)

            rows.append(
                DarpPoolRow(
                    pool_id=pool_id,
                    passenger_idx=r,
                    route_idx=int(rho),
                    leg_kind="first_mile",
                    origin=o_r,
                    destination=sb,
                    time=t_r,
                )
            )
            pool_id += 1

            first_mile_time = float(dm[o_r][sb]) if dm is not None and sb >= 0 else 0.0
            t_board = t_r + first_mile_time + td

            line_length = int(set_of_lines[rho][0])
            if line_length > 0 and lengths_travel_times is not None:
                segment_edges = max(0, opt.mt_drop_off_line_edge_index - opt.mt_pickup_line_edge_index)
                segment_time = float(lengths_travel_times[rho]) * (segment_edges / line_length)
            else:
                segment_time = 0.0
            t_unboard = t_board + segment_time

            rows.append(
                DarpPoolRow(
                    pool_id=pool_id,
                    passenger_idx=r,
                    route_idx=int(rho),
                    leg_kind="last_mile",
                    origin=su,
                    destination=d_r,
                    time=float(t_unboard),
                )
            )
            pool_id += 1

    return rows


def pool_key_to_pool_id_map(pool_rows: Sequence[DarpPoolRow]) -> Dict[Tuple[int, Optional[int], str], int]:
    m: Dict[Tuple[int, Optional[int], str], int] = {}
    for row in pool_rows:
        key = (row.passenger_idx, row.route_idx, row.leg_kind)
        if key in m:
            raise ValueError(f"Duplicate DARP pool key {key}")
        m[key] = row.pool_id
    return m


def fit_kmeans_pool_cluster_labels(
    line_instance: "lineplanning.instance.line_instance",
    pool_rows: Sequence[DarpPoolRow],
    feature_names: Tuple[str, ...],
    n_clusters: int,
    *,
    random_state: int,
    n_init: int,
) -> np.ndarray:
    """Return shape (len(pool_rows),) integer cluster label per pool row."""
    n_samples = len(pool_rows)
    if n_samples == 0:
        return np.zeros(0, dtype=int)
    X = _pool_rows_feature_matrix(line_instance, tuple(pool_rows), feature_names)
    k_req = max(1, int(n_clusters))
    k_eff = min(k_req, n_samples)
    if k_eff < k_req:
        logging.warning(
            "KMeans: reducing clusters from %s to %s (pool size=%s).",
            k_req,
            k_eff,
            n_samples,
        )
    km = KMeans(
        n_clusters=k_eff,
        random_state=int(random_state),
        n_init=int(n_init),
    )
    return km.fit_predict(X)


def select_darp_requests_from_pool(
    pool_rows: Sequence[DarpPoolRow],
    request_assignments: List[Tuple[str, Optional[int]]],
) -> Tuple[List[dict], Dict[int, int]]:
    """
    Same passenger iteration order and leg ordering as :func:`solution_to_darp_requests`.

    Returns exported DARP request dicts (sequential ``id`` 0..n-1) and ``exported_id -> pool_id``.
    """
    key_to_pid = pool_key_to_pool_id_map(pool_rows)
    by_id = {r.pool_id: r for r in pool_rows}

    darp_requests: List[dict] = []
    exported_to_pool: Dict[int, int] = {}
    export_id = 0
    nb_pass = len(request_assignments)

    for r in range(nb_pass):
        kind, line_idx = request_assignments[r]
        if kind == "rejected":
            continue
        if kind == "no_MT" or line_idx is None:
            pid = key_to_pid[(r, None, "no_mt")]
            row = by_id[pid]
            darp_requests.append(
                {
                    "id": export_id,
                    "original_request_id": r,
                    "origin": row.origin,
                    "destination": row.destination,
                    "time": row.time,
                }
            )
            exported_to_pool[export_id] = pid
            export_id += 1
            continue

        route = int(line_idx)
        pid_fm = key_to_pid[(r, route, "first_mile")]
        pid_lm = key_to_pid[(r, route, "last_mile")]
        row_fm = by_id[pid_fm]
        row_lm = by_id[pid_lm]

        darp_requests.append(
            {
                "id": export_id,
                "original_request_id": r,
                "origin": row_fm.origin,
                "destination": row_fm.destination,
                "time": row_fm.time,
            }
        )
        exported_to_pool[export_id] = pid_fm
        export_id += 1

        darp_requests.append(
            {
                "id": export_id,
                "original_request_id": r,
                "origin": row_lm.origin,
                "destination": row_lm.destination,
                "time": row_lm.time,
            }
        )
        exported_to_pool[export_id] = pid_lm
        export_id += 1

    return darp_requests, exported_to_pool


def write_darp_pool_export_map(path: Path, exported_to_pool: Dict[int, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [exported_to_pool[i] for i in range(len(exported_to_pool))]
    path.write_text(json.dumps({"export_pool_id_order": order}, indent=2), encoding="utf-8")


def load_darp_pool_export_map(path: Path) -> Dict[int, int]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    order = data["export_pool_id_order"]
    return {i: int(pid) for i, pid in enumerate(order)}


def _old_mod_leg_cost_on_instance(
    line_instance: "lineplanning.instance.line_instance",
    pool_row: DarpPoolRow,
) -> float:
    p = pool_row.passenger_idx
    if pool_row.leg_kind == "no_mt":
        opt = line_instance.direct_trip_options[p]
        return float(opt.first_mile_cost) + float(opt.last_mile_cost)
    assert pool_row.route_idx is not None
    opt = line_instance.optimal_trip_options[p][int(pool_row.route_idx)]
    if pool_row.leg_kind == "first_mile":
        return float(opt.first_mile_cost)
    if pool_row.leg_kind == "last_mile":
        return float(opt.last_mile_cost)
    raise AssertionError(f"unexpected leg_kind {pool_row.leg_kind!r}")


def _sanity_check_pool_export_matches_solution_to_darp(
    line_instance: "lineplanning.instance.line_instance",
    request_assignments: List[Tuple[str, Optional[int]]],
    pool_export_requests: List[dict],
    *,
    transfer_delay: float,
) -> None:
    legacy = solution_to_darp_requests(
        line_instance,
        request_assignments,
        request_times=None,
        transfer_delay=transfer_delay,
    )

    def norm_rows(rows: List[dict]) -> List[Tuple[Any, ...]]:
        out = []
        for row in rows:
            out.append(
                (
                    int(row["original_request_id"]),
                    int(row["origin"]),
                    int(row["destination"]),
                    float(row["time"]),
                )
            )
        return out

    if norm_rows(legacy) != norm_rows(pool_export_requests):
        raise AssertionError(
            "Pool export differs from solution_to_darp_requests:\n"
            f"legacy={norm_rows(legacy)!r}\npool={norm_rows(pool_export_requests)!r}"
        )


def _parse_darp_vehicles_and_max_delay(
    raw: Dict[str, Any],
    experiment_yaml_path: Path,
) -> Tuple[int, int]:
    """
    Read DARP instance-style keys from experiment YAML (Ridesharing_DARP_instances layout):

    - ``vehicles.vehicle_capacity`` (default 5 if the ``vehicles`` mapping is absent or omits the key).
    - ``max_travel_time_delay`` with ``mode: absolute`` and numeric ``seconds`` (defaults to 300 seconds
      when the block is absent or empty).
    """
    vehicles = raw.get("vehicles") or {}
    darp_vehicle_capacity = int(vehicles.get("vehicle_capacity", 5))

    mtd = raw.get("max_travel_time_delay")
    if isinstance(mtd, dict) and mtd:
        mode_raw = mtd.get("mode", "absolute")
        mode = str(mode_raw).strip().lower()
        if mode != "absolute":
            raise ValueError(
                f"{experiment_yaml_path}: max_travel_time_delay.mode must be 'absolute', got {mode_raw!r}."
            )
        sec_raw = mtd.get("seconds")
        if sec_raw is None:
            raise ValueError(
                f"{experiment_yaml_path}: max_travel_time_delay.seconds is required when "
                "max_travel_time_delay is set."
            )
        max_travel_time_delay_seconds = int(sec_raw)
    else:
        max_travel_time_delay_seconds = 300

    return darp_vehicle_capacity, max_travel_time_delay_seconds


def _parse_yaml_bool(raw: object, default: bool = False) -> bool:
    """Coerce YAML values to bool (native bool and common string forms)."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


@dataclass(frozen=True)
class ModAwareLineSelectionConfig:
    """Resolved settings from a MoD-aware ``experiment.yaml``."""

    experiment_yaml_path: Path
    results_dir: Path
    instance_paths: LinePlanningInstancePaths
    initial_mod_cost_scale: float
    iterations: int
    solver_time_limit: float
    rejection_cost: float
    use_request_line_valid_inequalities: bool
    reuse_model: bool
    cost_coefficient: float
    max_frequency: int
    capacity: int
    maximum_detour: Any
    line_mod_aggregate_prune: bool
    darp_benchmark_executable: Path
    transfer_delay: float
    darp_vehicle_capacity: int
    max_travel_time_delay_seconds: int
    mod_cost_recomputation_strategy: str
    mod_cost_smoothing_strategy: str
    mod_cost_under_relaxation_alpha: float
    mod_cost_recomputation_clusters: Optional[int]
    mod_cost_recomputation_features: Optional[Tuple[str, ...]]
    kmeans_random_state: int
    kmeans_n_init: int


def load_mod_aware_line_selection_config(experiment_yaml_path: Path) -> ModAwareLineSelectionConfig:
    """
    Load MoD-aware line selection settings from an experiment YAML file.

    Required:
    - ``instance``: path to the line-planning instance ``config.yaml`` (same as ``run_experiment``).

    Optional ``darp.benchmark_executable``: path to the DARP-benchmark binary; if omitted, uses
    module default :data:`DARP_BENCHMARK_PATH` (from ``DARP_BENCHMARK_EXECUTABLE`` when set).

    Optional ``results_dir``: output root (defaults to the experiment file's directory); see
    :func:`lineplanning.instance_config.resolve_results_dir`.

    Optional ``initial_mod_cost_scale`` (default 1.0): scales stored MoD costs after load.

    Optional ``darp.transfer_delay`` (seconds, default 0): extra delay δ_transfer when building DARP
    request times for first/last mile.

    DARP instance fields (same shape as Ridesharing_DARP_instances ``config.yaml``): optional
    ``vehicles.vehicle_capacity`` and ``max_travel_time_delay`` with ``mode: absolute`` and
    ``seconds: <int>``.

    MoD cost update: ``mod_cost_recomputation`` with ``strategy`` and optional nested ``smoothing``
    (``strategy``, ``under_relaxation_alpha``).

    Other keys default to the previous script defaults documented in the module docstring.
    """
    experiment_yaml_path = Path(experiment_yaml_path).resolve()
    raw = lineplanning.instance_config.load_experiment_yaml(experiment_yaml_path)
    inst_path = lineplanning.instance_config.resolve_instance_config_path(experiment_yaml_path, raw)
    instance_paths = lineplanning.instance_config.load_line_planning_instance_config(inst_path)
    results_dir = lineplanning.instance_config.resolve_results_dir(experiment_yaml_path, raw)

    solver = raw.get("solver") or {}
    mt = raw.get("mass_transport") or {}
    darp = raw.get("darp") or {}
    mod_recomputation = raw.get("mod_cost_recomputation") or {}

    initial_mod_cost_scale = _parse_initial_mod_cost_scale(raw, experiment_yaml_path)

    iterations_raw = raw.get("iterations", 30)
    try:
        iterations = int(iterations_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid iterations in {experiment_yaml_path}: expected int, got {iterations_raw!r}."
        ) from exc
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1 in {experiment_yaml_path}, got {iterations}.")

    solver_time_limit = float(solver.get("time_limit", 1500))
    rejection_cost_cfg = float(solver.get("rejection_cost", 1000) or 0)
    use_request_line_valid_inequalities = _parse_yaml_bool(
        solver.get("use_request_line_valid_inequalities"), default=True
    )
    reuse_model = _parse_yaml_bool(solver.get("reuse_model"), default=True)

    cost_coefficient = float(mt.get("cost_coefficient", 1.0))
    max_frequency_cfg = int(mt.get("max_frequency", 20))
    capacity = int(mt.get("capacity", 30))
    maximum_detour = mt.get("maximum_detour", 3)
    line_mod_aggregate_prune = _parse_yaml_bool(mt.get("line_mod_aggregate_prune"), default=False)

    darp_exe = darp.get("benchmark_executable")
    if darp_exe is None or (isinstance(darp_exe, str) and not str(darp_exe).strip()):
        # Same default as module-level ``DARP_BENCHMARK_PATH`` (``DARP_BENCHMARK_EXECUTABLE`` env).
        darp_benchmark_executable = Path(DARP_BENCHMARK_PATH).resolve()
    else:
        darp_benchmark_executable = Path(str(darp_exe).strip()).expanduser()
        if not darp_benchmark_executable.is_absolute():
            darp_benchmark_executable = (experiment_yaml_path.parent / darp_benchmark_executable).resolve()
        else:
            darp_benchmark_executable = darp_benchmark_executable.resolve()

    transfer_delay = float(darp.get("transfer_delay", 0))

    darp_vehicle_capacity, max_travel_time_delay_seconds = _parse_darp_vehicles_and_max_delay(
        raw,
        experiment_yaml_path,
    )

    mod_cost_recomputation_strategy = str(
        mod_recomputation.get("strategy", "aggregate_original_requests")
    ).strip()

    smoothing_block = mod_recomputation.get("smoothing")
    if not isinstance(smoothing_block, dict):
        smoothing_block = {}
    mod_cost_smoothing_strategy = str(smoothing_block.get("strategy", "none")).strip()
    mod_cost_under_relaxation_alpha = float(smoothing_block.get("under_relaxation_alpha", 0.3))

    clusters_raw = mod_recomputation.get("clusters")
    features_raw = mod_recomputation.get("features")
    mod_cost_recomputation_clusters: Optional[int] = None
    mod_cost_recomputation_features: Optional[Tuple[str, ...]] = None
    if clusters_raw is not None:
        mod_cost_recomputation_clusters = int(clusters_raw)
    if features_raw is not None:
        if isinstance(features_raw, (list, tuple)):
            mod_cost_recomputation_features = tuple(str(x).strip() for x in features_raw if str(x).strip())
        else:
            raise ValueError(
                f"{experiment_yaml_path}: mod_cost_recomputation.features must be a list, got {type(features_raw).__name__}."
            )

    kmeans_random_state = int(mod_recomputation.get("random_state", 0))
    kmeans_n_init = int(mod_recomputation.get("n_init", 10))

    if mod_cost_recomputation_strategy == "clustered_avg":
        if mod_cost_recomputation_clusters is None or mod_cost_recomputation_clusters < 1:
            raise ValueError(
                f"{experiment_yaml_path}: mod_cost_recomputation.strategy=clustered_avg requires "
                "mod_cost_recomputation.clusters (int >= 1)."
            )
        if not mod_cost_recomputation_features:
            raise ValueError(
                f"{experiment_yaml_path}: mod_cost_recomputation.strategy=clustered_avg requires "
                "a non-empty mod_cost_recomputation.features list."
            )

    return ModAwareLineSelectionConfig(
        experiment_yaml_path=experiment_yaml_path,
        results_dir=results_dir,
        instance_paths=instance_paths,
        initial_mod_cost_scale=initial_mod_cost_scale,
        iterations=iterations,
        solver_time_limit=solver_time_limit,
        rejection_cost=rejection_cost_cfg,
        use_request_line_valid_inequalities=use_request_line_valid_inequalities,
        reuse_model=reuse_model,
        cost_coefficient=cost_coefficient,
        max_frequency=max_frequency_cfg,
        capacity=capacity,
        maximum_detour=maximum_detour,
        line_mod_aggregate_prune=line_mod_aggregate_prune,
        darp_benchmark_executable=darp_benchmark_executable,
        transfer_delay=transfer_delay,
        darp_vehicle_capacity=darp_vehicle_capacity,
        max_travel_time_delay_seconds=max_travel_time_delay_seconds,
        mod_cost_recomputation_strategy=mod_cost_recomputation_strategy,
        mod_cost_smoothing_strategy=mod_cost_smoothing_strategy,
        mod_cost_under_relaxation_alpha=mod_cost_under_relaxation_alpha,
        mod_cost_recomputation_clusters=mod_cost_recomputation_clusters,
        mod_cost_recomputation_features=mod_cost_recomputation_features,
        kmeans_random_state=kmeans_random_state,
        kmeans_n_init=kmeans_n_init,
    )


def _load_demand_and_dm_from_instance_config(instance_dir_path: Path) -> Tuple[Path, Path]:
    demand_path, dm_path, _cfg = _load_line_planning_paths_from_instance_dir(instance_dir_path)
    return demand_path, dm_path


def solution_to_darp_requests(
    line_instance: "lineplanning.instance.line_instance",
    request_assignments: List[Tuple[str, Optional[int]]],
    request_times: Optional[List[Union[int, float]]] = None,
    transfer_delay: Union[int, float] = 0,
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
        t_board = t_r + first_mile_time + transfer_delay

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
    
    logging.info("Scaling MoD costs in the model by %s", factor)

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
    clustered_avg_state: Optional[ClusteredAvgState] = None


ModCostRecomputeStrategy = Callable[[ModCostRecomputeContext], Dict[int, Tuple[float, float]]]


@dataclass(frozen=True)
class ModCostSmoothingContext:
    """
    Inputs for MoD cost smoothing applied **after** a recompute strategy and **before**
    ``update_mod_costs``.

    ``previous`` must reflect coefficients already stored on the instance for the current
    assignment pattern *after* the recompute strategy returns (e.g. after global rescaling),
    so that ``previous`` and ``proposed`` live in the same scale.

    Extra fields are available for future smoothing rules without changing call sites.
    """

    proposed: Dict[int, Tuple[float, float]]
    previous: Dict[int, Tuple[float, float]]
    line_instance: "lineplanning.instance.line_instance"
    request_assignments: List[Tuple[str, Optional[int]]]


ModCostSmoothingStrategy = Callable[[ModCostSmoothingContext], Dict[int, Tuple[float, float]]]


def smoothing__none(ctx: ModCostSmoothingContext) -> Dict[int, Tuple[float, float]]:
    """Pass through recompute output unchanged."""
    return dict(ctx.proposed)


def make_smoothing_under_relaxation(alpha: float) -> ModCostSmoothingStrategy:
    """
    Under-relaxation on the (first_mile, last_mile) pair per served request::

        c_new = (1 - α) * c_DARP + α * c_old

    with ``c_DARP`` = proposed and ``c_old`` = previous (from context), component-wise.
    ``alpha`` is read from experiment YAML ``mod_cost_recomputation.smoothing.under_relaxation_alpha``.
    """
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not (0.0 <= alpha_f <= 1.0):
        raise ValueError(
            f"mod_cost_recomputation.smoothing.under_relaxation_alpha must be finite and in [0, 1], got {alpha!r}."
        )

    def smoothing_under_relaxation(ctx: ModCostSmoothingContext) -> Dict[int, Tuple[float, float]]:
        if ctx.proposed.keys() != ctx.previous.keys():
            raise ValueError(
                "Smoothing requires proposed and previous cost dicts with identical keys; "
                f"proposed={sorted(ctx.proposed.keys())} previous={sorted(ctx.previous.keys())}"
            )
        out: Dict[int, Tuple[float, float]] = {}
        oma = 1.0 - alpha_f
        for k, (pf, pl) in ctx.proposed.items():
            of, ol = ctx.previous[k]
            out[k] = (oma * float(pf) + alpha_f * float(of), oma * float(pl) + alpha_f * float(ol))
        return out

    return smoothing_under_relaxation


def resolve_mod_cost_smoothing_strategy(
    name: str,
    *,
    under_relaxation_alpha: float,
) -> ModCostSmoothingStrategy:
    """Build smoothing from experiment YAML ``mod_cost_recomputation.smoothing.strategy``."""
    key = str(name).strip()
    if key == "none":
        return smoothing__none
    if key == "under_relaxation":
        return make_smoothing_under_relaxation(under_relaxation_alpha)
    raise KeyError(
        f"Unknown mod_cost_recomputation.smoothing.strategy={name!r}. Available: none, under_relaxation"
    )


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


def _apply_cluster_factors_to_all_mod_costs(
    line_instance: "lineplanning.instance.line_instance",
    pool_rows: Sequence[DarpPoolRow],
    pool_id_to_cluster: Dict[int, int],
    factor_per_cluster: Dict[int, float],
) -> None:
    """Multiply stored MoD leg costs by per-cluster factors (FM/LM may differ for line options)."""
    key_to_pid = pool_key_to_pool_id_map(pool_rows)

    for p in range(line_instance.nb_pass):
        pid = key_to_pid[(p, None, "no_mt")]
        c = pool_id_to_cluster[pid]
        f = float(factor_per_cluster.get(c, 1.0))
        if f == 1.0:
            continue
        opt = line_instance.direct_trip_options[p]
        line_instance.direct_trip_options[p] = opt._replace(
            first_mile_cost=float(opt.first_mile_cost) * f,
            last_mile_cost=float(opt.last_mile_cost) * f,
        )

    for p in range(line_instance.nb_pass):
        for rho, opt in list(line_instance.optimal_trip_options[p].items()):
            irho = int(rho)
            pid_fm = key_to_pid[(p, irho, "first_mile")]
            pid_lm = key_to_pid[(p, irho, "last_mile")]
            cf = float(factor_per_cluster.get(pool_id_to_cluster[pid_fm], 1.0))
            cl = float(factor_per_cluster.get(pool_id_to_cluster[pid_lm], 1.0))
            if cf == 1.0 and cl == 1.0:
                continue
            line_instance.optimal_trip_options[p][rho] = opt._replace(
                first_mile_cost=float(opt.first_mile_cost) * cf,
                last_mile_cost=float(opt.last_mile_cost) * cl,
            )


def recompute_costs__clustered_avg(ctx: ModCostRecomputeContext) -> Dict[int, Tuple[float, float]]:
    """
    Per-cluster rescaling of MoD costs using average (cost / leg travel time) from DARP vs instance.

    Requires :attr:`ModCostRecomputeContext.clustered_avg_state` (pool built once, K-means once).
    """
    aux = ctx.clustered_avg_state
    if aux is None:
        raise ValueError("clustered_avg recomputation requires ModCostRecomputeContext.clustered_avg_state")

    pool_by_id = {r.pool_id: r for r in aux.pool_rows}
    key_to_pid = pool_key_to_pool_id_map(aux.pool_rows)

    new_ratios: Dict[int, List[float]] = {}
    old_ratios: Dict[int, List[float]] = {}

    for exp_id, pool_id in aux.exported_id_to_pool_id.items():
        row = pool_by_id[pool_id]
        cluster = aux.pool_id_to_cluster[pool_id]
        tt = _pool_feature_travel_time(ctx.line_instance, row)
        if tt <= 0.0:
            continue
        new_cost = sum(float(x) for x in ctx.darp_request_leg_costs[int(exp_id)])
        old_cost = _old_mod_leg_cost_on_instance(ctx.line_instance, row)
        new_ratios.setdefault(cluster, []).append(new_cost / tt)
        old_ratios.setdefault(cluster, []).append(old_cost / tt)

    factor_per_cluster: Dict[int, float] = {}
    for cluster in sorted(set(aux.pool_id_to_cluster.values())):
        nv = new_ratios.get(cluster, [])
        ov = old_ratios.get(cluster, [])
        if not nv or not ov:
            factor_per_cluster[cluster] = 1.0
            logging.warning(
                "MoD clustered_avg: cluster %s has insufficient DARP ratios (new=%s old=%s); factor=1.0.",
                cluster,
                len(nv),
                len(ov),
            )
            continue
        new_avg = sum(nv) / len(nv)
        old_avg = sum(ov) / len(ov)
        if new_avg <= 0.0 or old_avg <= 0.0:
            logging.warning(
                "MoD clustered_avg: cluster %s invalid averages (new_avg=%s old_avg=%s); factor=1.0.",
                cluster,
                new_avg,
                old_avg,
            )
            factor_per_cluster[cluster] = 1.0
            continue
        factor_per_cluster[cluster] = new_avg / old_avg

    logging.info("MoD clustered_avg factors by cluster: %s", factor_per_cluster)

    _apply_cluster_factors_to_all_mod_costs(
        ctx.line_instance,
        aux.pool_rows,
        aux.pool_id_to_cluster,
        factor_per_cluster,
    )

    raw_costs = aggregate_mod_costs_for_original_requests(
        ctx.darp_request_leg_costs,
        ctx.darp_requests,
        ctx.request_assignments,
    )
    out: Dict[int, Tuple[float, float]] = {}
    for orig_id, (fm, lm) in raw_costs.items():
        kind, rho = ctx.request_assignments[orig_id]
        if kind == "rejected":
            continue
        if kind == "no_MT" or rho is None:
            pid = key_to_pid[(orig_id, None, "no_mt")]
            f = float(factor_per_cluster.get(aux.pool_id_to_cluster[pid], 1.0))
            out[orig_id] = (float(fm) * f, float(lm) * f)
        else:
            irho = int(rho)
            pid_fm = key_to_pid[(orig_id, irho, "first_mile")]
            pid_lm = key_to_pid[(orig_id, irho, "last_mile")]
            ff = float(factor_per_cluster.get(aux.pool_id_to_cluster[pid_fm], 1.0))
            fl = float(factor_per_cluster.get(aux.pool_id_to_cluster[pid_lm], 1.0))
            out[orig_id] = (float(fm) * ff, float(lm) * fl)

    return out


MOD_COST_RECOMPUTATION_STRATEGIES: Dict[str, ModCostRecomputeStrategy] = {
    # Baseline / reference implementation (current script behavior)
    "aggregate_original_requests": recompute_costs__aggregate_original_requests,
    # First alternative strategy (requested)
    "rescale_avg_cost_per_traveltime_and_request": (
        recompute_costs__rescale_avg_cost_per_traveltime_and_request
    ),
    "clustered_avg": recompute_costs__clustered_avg,
}

MOD_COST_SNAPSHOT_FILENAME = "mod_cost_snapshot.csv"
MOD_COST_INITIAL_FILENAME = "mod_costs_initial.csv"

_MOD_COST_CSV_FIELDNAMES = (
    "passenger_idx",
    "line_idx",
    "first_mile_cost",
    "last_mile_cost",
    "mt_cost",
)


def read_mod_costs_csv(path: Path) -> List[Dict[str, Any]]:
    """Load rows written by :func:`export_mod_costs_csv`."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        missing = set(_MOD_COST_CSV_FIELDNAMES) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} missing columns {sorted(missing)}; "
                f"expected {list(_MOD_COST_CSV_FIELDNAMES)}"
            )
        rows: List[Dict[str, Any]] = []
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "passenger_idx": int(row["passenger_idx"]),
                        "line_idx": int(row["line_idx"]),
                        "first_mile_cost": float(row["first_mile_cost"]),
                        "last_mile_cost": float(row["last_mile_cost"]),
                        "mt_cost": float(row["mt_cost"]),
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} line {line_no}: invalid row {row!r}") from exc
        return rows


def apply_mod_cost_rows_to_instance(
    line_inst: "lineplanning.instance.line_instance",
    rows: List[Dict[str, Any]],
) -> None:
    """
    Apply per-option MoD cost columns from CSV rows.
    ``line_idx == -1`` denotes the direct (no line) option for that passenger; otherwise
    ``line_idx`` is a candidate line index, matching ``preprocessing`` cache rows.
    """
    for row in rows:
        p = int(row["passenger_idx"])
        rho = int(row["line_idx"])
        fm = float(row["first_mile_cost"])
        lm = float(row["last_mile_cost"])
        mc = float(row["mt_cost"])
        if rho < 0:
            if p < 0 or p >= len(line_inst.direct_trip_options):
                continue
            t = line_inst.direct_trip_options[p]
            line_inst.direct_trip_options[p] = t._replace(
                first_mile_cost=fm,
                last_mile_cost=lm,
                mt_cost=mc,
            )
        else:
            if p < 0 or p >= len(line_inst.optimal_trip_options):
                continue
            if rho not in line_inst.optimal_trip_options[p]:
                continue
            t = line_inst.optimal_trip_options[p][rho]
            line_inst.optimal_trip_options[p][rho] = t._replace(
                first_mile_cost=fm,
                last_mile_cost=lm,
                mt_cost=mc,
            )


def export_mod_costs_csv(
    line_inst: "lineplanning.instance.line_instance",
    path: Path,
) -> None:
    """
    Export only MoD cost coefficients that can change across iterations.

    Rows use the same ``(passenger_idx, line_idx)`` keys as preprocessing trip-option CSVs;
    direct options use ``line_idx=-1`` (not stored in preprocessing cache). Static trip
    geometry and ``value`` stay in preprocessing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for p in range(line_inst.nb_pass):
        opt = line_inst.direct_trip_options[p]
        rows.append(
            {
                "passenger_idx": p,
                "line_idx": -1,
                "first_mile_cost": opt.first_mile_cost,
                "last_mile_cost": opt.last_mile_cost,
                "mt_cost": opt.mt_cost,
            }
        )
    for p in range(line_inst.nb_pass):
        for rho in sorted(line_inst.optimal_trip_options[p].keys()):
            opt = line_inst.optimal_trip_options[p][rho]
            rows.append(
                {
                    "passenger_idx": p,
                    "line_idx": rho,
                    "first_mile_cost": opt.first_mile_cost,
                    "last_mile_cost": opt.last_mile_cost,
                    "mt_cost": opt.mt_cost,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_MOD_COST_CSV_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Wrote MoD costs CSV %s (%d rows)", path, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_config",
        type=Path,
        help="Path to experiment.yaml (MoD-aware configuration; see module docstring).",
    )
    args = parser.parse_args()

    cfg = load_mod_aware_line_selection_config(args.experiment_config)

    inst = cfg.instance_paths
    demand_file = inst.demand_file
    dm_file = inst.dm_file
    preprocessing_dir = inst.config_path.parent / "preprocessing"

    results_dir_path = cfg.results_dir
    results_dir_path.mkdir(parents=True, exist_ok=True)

    setup_file_logging(results_dir_path)
    logging.info("Loaded experiment configuration from %s", cfg.experiment_yaml_path)
    logging.info("Results directory: %s", cfg.results_dir)
    logging.info("DARP benchmark executable: %s", cfg.darp_benchmark_executable)

    prune_rej = cfg.rejection_cost if cfg.rejection_cost > 0 else None
    line_inst = lineplanning.instance.line_instance(
        candidate_lines_file=inst.lines_file,
        capacity=cfg.capacity,
        maximum_detour=cfg.maximum_detour,
        demand_file=demand_file,
        preprocessing_dir=preprocessing_dir,
        dm_file=dm_file,
        line_mod_aggregate_prune=cfg.line_mod_aggregate_prune,
        line_mod_aggregate_prune_cost_coefficient=cfg.cost_coefficient,
        line_mod_aggregate_prune_rejection_cost=prune_rej,
    )

    if cfg.initial_mod_cost_scale != 1.0:
        logging.info(
            "Applying initial_mod_cost_scale=%s from experiment config %s (direct + all line MoD options)",
            cfg.initial_mod_cost_scale,
            cfg.experiment_yaml_path,
        )
        _scale_all_mod_costs_in_model(line_inst, cfg.initial_mod_cost_scale)

    solver = lineplanning.line_planning.LinePlanningSolver(
        line_inst,
        time_limit=cfg.solver_time_limit,
        cost_coefficient=cfg.cost_coefficient,
        max_frequency=cfg.max_frequency,
    )

    pool_rows: List[DarpPoolRow] = []
    pool_id_to_cluster: Dict[int, int] = {}
    if cfg.mod_cost_recomputation_strategy == "clustered_avg":
        assert cfg.mod_cost_recomputation_clusters is not None
        assert cfg.mod_cost_recomputation_features is not None
        pool_rows = build_darp_request_pool(
            line_inst,
            request_times=None,
            transfer_delay=cfg.transfer_delay,
        )
        expected_pool = 0
        for r in range(line_inst.nb_pass):
            expected_pool += 1
            for rho in line_inst.optimal_trip_options[r]:
                if line_inst.trip_option_on_line(r, rho) is not None:
                    expected_pool += 2
        if len(pool_rows) != expected_pool:
            raise RuntimeError(
                f"DARP pool row count mismatch: got {len(pool_rows)}, expected {expected_pool}"
            )
        labels = fit_kmeans_pool_cluster_labels(
            line_inst,
            pool_rows,
            cfg.mod_cost_recomputation_features,
            cfg.mod_cost_recomputation_clusters,
            random_state=cfg.kmeans_random_state,
            n_init=cfg.kmeans_n_init,
        )
        pool_id_to_cluster = {row.pool_id: int(labels[i]) for i, row in enumerate(pool_rows)}
        logging.info(
            "DARP request pool: %s legs, %s cluster labels (requested k=%s, features=%s)",
            len(pool_rows),
            len(set(pool_id_to_cluster.values())),
            cfg.mod_cost_recomputation_clusters,
            cfg.mod_cost_recomputation_features,
        )

    instance_size_label = lineplanning.line_planning.get_instance_size_label(str(demand_file))

    export_mod_costs_csv(line_inst, results_dir_path / MOD_COST_INITIAL_FILENAME)

    recompute_strategy = MOD_COST_RECOMPUTATION_STRATEGIES.get(cfg.mod_cost_recomputation_strategy)
    if recompute_strategy is None:
        raise KeyError(
            f"Unknown mod_cost_recomputation.strategy={cfg.mod_cost_recomputation_strategy!r}. "
            f"Available: {sorted(MOD_COST_RECOMPUTATION_STRATEGIES.keys())}"
        )
    smoother = resolve_mod_cost_smoothing_strategy(
        cfg.mod_cost_smoothing_strategy,
        under_relaxation_alpha=cfg.mod_cost_under_relaxation_alpha,
    )

    try:
        for i in range(cfg.iterations):
            logging.info("Iteration %s of %s", i + 1, cfg.iterations)

            results_dir_path_per_iteration = results_dir_path / f"iteration_{i+1}"
            results_dir_path_per_iteration.mkdir(parents=True, exist_ok=True)

            # Check if DARP input files already exist; if so, skip to step 2.2
            requests_csv_path = results_dir_path_per_iteration / "requests.csv"
            passenger_assignments_csv_path = results_dir_path_per_iteration / "passenger_assignments.csv"
            experiment_config_path = results_dir_path_per_iteration / "experiment_ih.yaml"

            exported_to_pool: Optional[Dict[int, int]] = None

            if requests_csv_path.exists() and passenger_assignments_csv_path.exists() and experiment_config_path.exists():
                logging.info(
                    "Iteration %s: DARP input files exist, skipping ILP and loading from files",
                    i + 1,
                )
                darp_requests = load_darp_requests_csv(requests_csv_path)
                request_assignments = load_request_assignments_csv(
                    passenger_assignments_csv_path,
                )
                if cfg.mod_cost_recomputation_strategy == "clustered_avg":
                    map_path = results_dir_path_per_iteration / DARP_POOL_EXPORT_MAP_FILENAME
                    if map_path.is_file():
                        exported_to_pool = load_darp_pool_export_map(map_path)
                    else:
                        _, exported_to_pool = select_darp_requests_from_pool(
                            pool_rows,
                            request_assignments,
                        )
                        logging.warning(
                            "Iteration %s: missing %s; rebuilt export mapping from assignments",
                            i + 1,
                            DARP_POOL_EXPORT_MAP_FILENAME,
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
                        max_route_frequency=cfg.max_frequency,
                        rejection_cost=cfg.rejection_cost,
                        use_request_line_valid_inequalities=cfg.use_request_line_valid_inequalities,
                        reuse_model=cfg.reuse_model,
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
                    "experiment_config": str(cfg.experiment_yaml_path),
                    "max_route_frequency": cfg.max_frequency,
                    "solver_time_limit": cfg.solver_time_limit,
                    "rejection_cost": cfg.rejection_cost,
                    "mod_cost_recomputation_strategy": cfg.mod_cost_recomputation_strategy,
                    "mod_cost_smoothing_strategy": cfg.mod_cost_smoothing_strategy,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                results_file = results_dir_path_per_iteration / "metrics.json"
                results_file.write_text(json.dumps(results_payload, indent=2))

                # 2. DARP — export requests.csv (pool selection for clustered_avg, else legacy builder)
                if cfg.mod_cost_recomputation_strategy == "clustered_avg":
                    darp_requests, exported_to_pool = select_darp_requests_from_pool(
                        pool_rows,
                        request_assignments,
                    )
                    _sanity_check_pool_export_matches_solution_to_darp(
                        line_inst,
                        request_assignments,
                        darp_requests,
                        transfer_delay=cfg.transfer_delay,
                    )
                    write_darp_pool_export_map(
                        results_dir_path_per_iteration / DARP_POOL_EXPORT_MAP_FILENAME,
                        exported_to_pool,
                    )
                else:
                    darp_requests = solution_to_darp_requests(
                        line_inst,
                        request_assignments,
                        request_times=None,
                        transfer_delay=cfg.transfer_delay,
                    )
                write_darp_requests_csv(darp_requests, requests_csv_path, time_format="seconds")
                write_darp_vehicles_csv(
                    darp_requests,
                    results_dir_path_per_iteration / "vehicles.csv",
                    capacity=cfg.darp_vehicle_capacity,
                    time_format="seconds",
                )
                write_darp_config_yaml(
                    output_dir=results_dir_path_per_iteration,
                    dm_filepath=dm_file,
                    max_travel_time_delay_seconds=cfg.max_travel_time_delay_seconds,
                    vehicle_capacity=cfg.darp_vehicle_capacity,
                )

            # 2.2 call DARP solver
            darpbenchmark.experiments.run_experiment_using_config(
                experiment_config_path,
                executable_path=cfg.darp_benchmark_executable,
            )

            # 3. Recompute the MoD cost estimates (section 4.3.2)
            solution_path = results_dir_path_per_iteration / "config.yaml-solution.json"
            darp_solution = darpinstances.inout.load_json(solution_path)

            darp_request_leg_costs = compute_per_darp_request_costs(
                darp_solution,
                darp_requests,
                request_assignments,
            )

            clustered_state: Optional[ClusteredAvgState] = None
            if cfg.mod_cost_recomputation_strategy == "clustered_avg":
                if exported_to_pool is None:
                    raise RuntimeError(
                        "clustered_avg requires exported_id→pool_id mapping (export map or pool selection)"
                    )
                clustered_state = ClusteredAvgState(
                    pool_rows=tuple(pool_rows),
                    pool_id_to_cluster=dict(pool_id_to_cluster),
                    exported_id_to_pool_id=dict(exported_to_pool),
                    feature_names=tuple(cfg.mod_cost_recomputation_features or ()),
                    n_clusters_requested=int(cfg.mod_cost_recomputation_clusters or 0),
                )

            ctx = ModCostRecomputeContext(
                line_instance=line_inst,
                request_assignments=request_assignments,
                darp_request_leg_costs=darp_request_leg_costs,
                darp_requests=darp_requests,
                clustered_avg_state=clustered_state,
            )
            new_mod_costs = recompute_strategy(ctx)

            previous_mod_costs = _get_current_costs_for_assignments(
                line_inst, request_assignments
            )
            smooth_ctx = ModCostSmoothingContext(
                proposed=new_mod_costs,
                previous=previous_mod_costs,
                line_instance=line_inst,
                request_assignments=request_assignments,
            )
            final_mod_costs = smoother(smooth_ctx)

            solver.update_mod_costs(final_mod_costs, request_assignments)
            logging.info("Iteration %s: updated MoD costs for %s requests", i + 1, len(new_mod_costs))
            export_mod_costs_csv(
                line_inst,
                results_dir_path_per_iteration / MOD_COST_SNAPSHOT_FILENAME,
            )
    except Exception:
        logging.exception("Uncaught error in MoD-aware line selection script")
        raise


if __name__ == "__main__":
    main()

