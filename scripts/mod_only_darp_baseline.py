"""
Run DARP-benchmark on a **MoD-only** baseline: one direct trip per **original** line-planning
passenger (origin → destination from demand). No mass-transit legs and no ``line_instance`` /
candidate-lines / preprocessing logic.

Uses the same ``requests.csv`` / ``vehicles.csv`` writers and DARP ``config.yaml`` /
``experiment_ih.yaml`` layout as ``MoD-aware_line_selection.py``. Settings use the same defaults
via ``build_mod_aware_line_selection_config`` (no separate experiment YAML).

**CLI**

- ``--instance-dir``: line-planning instance folder (contains ``config.yaml``); used only to
  resolve ``demand`` and ``dm_filepath`` paths.
- ``--experiment-dir``: directory where DARP inputs and ``experiment_ih.yaml`` are written and
  where DARP-benchmark runs (``outdir: .``). The distance matrix path in ``config.yaml`` is
  relative to this directory when possible, otherwise an absolute path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

import darpbenchmark.experiments
import lineplanning.instance_config


def _load_mod_aware_module() -> Any:
    path = Path(__file__).resolve().parent / "MoD-aware_line_selection.py"
    spec = importlib.util.spec_from_file_location("mod_aware_line_selection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_instance_dir(instance_dir: Path) -> Path:
    d = Path(instance_dir).resolve()
    if not d.is_dir():
        raise NotADirectoryError(f"instance-dir must be a directory: {d}")
    cfg = d / "config.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"No config.yaml in instance-dir: {d}")
    return d


def _load_original_requests(demand_file: Path) -> List[Tuple[int, int, float]]:
    """
    Load (origin, destination, time) per passenger from the line-planning demand file.

    Same rules as ``lineplanning.instance.line_instance.manhattan_instance``: ``.txt`` rows are
    whitespace-split; otherwise CSV is read via strict ``requests.csv`` columns.
    """
    from lineplanning.instance import _load_demand_from_csv

    demand_file = Path(demand_file)
    if demand_file.suffix.lower() == ".txt":
        with demand_file.open("r", encoding="utf-8") as f:
            my_list = [line.split() for line in f if line.strip()]
        requests_arr = None
    else:
        demand_df = _load_demand_from_csv(demand_file)
        requests_arr = demand_df.loc[:, ["origin", "destination", "time"]].to_numpy(copy=False)
        my_list = []

    requests: List[Tuple[int, int, float]] = []
    if requests_arr is not None:
        for o, d, t in requests_arr:
            requests.append((int(o), int(d), float(t)))
    else:
        for line_no, row in enumerate(my_list, start=1):
            if len(row) != 3:
                raise ValueError(
                    f"Demand text file {demand_file} line {line_no} must contain origin, destination and time"
                )
            o = int(float(row[0].strip()))
            d = int(float(row[1].strip()))
            t = float(row[2].strip())
            requests.append((o, d, t))
    return requests


def _original_requests_to_mod_darp_requests(
    requests: List[Tuple[int, int, float]],
) -> List[dict]:
    """
    One DARP row per original passenger — same dict shape as
    ``line_planning_original_requests_as_mod_darp_requests`` in ``MoD-aware_line_selection.py``.
    """
    out: List[dict] = []
    for r, (o_r, d_r, t_r) in enumerate(requests):
        out.append(
            {
                "id": r,
                "original_request_id": r,
                "origin": o_r,
                "destination": d_r,
                "time": float(t_r),
            }
        )
    return out


def _dm_path_for_darp_config(dm_file: Path, experiment_dir: Path) -> str:
    """Path string for DARP ``config.yaml``: relative to ``experiment_dir`` when possible."""
    dm_r = dm_file.resolve()
    exp_r = experiment_dir.resolve()
    try:
        return Path(os.path.relpath(dm_r, exp_r)).as_posix()
    except ValueError:
        return dm_r.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-dir",
        type=Path,
        required=True,
        help="Path to the line-planning instance directory (contains config.yaml).",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory for DARP requests/vehicles/config, experiment_ih.yaml, and benchmark output.",
    )
    args = parser.parse_args()

    instance_dir = _resolve_instance_dir(args.instance_dir)
    instance_config = instance_dir / "config.yaml"
    experiment_dir = Path(args.experiment_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    mod = _load_mod_aware_module()
    raw: Dict[str, Any] = {"instance": str(instance_config.resolve())}
    cfg = mod.build_mod_aware_line_selection_config(raw, instance_config)

    inst = lineplanning.instance_config.load_line_planning_instance_config(instance_config)
    dm_path_str = _dm_path_for_darp_config(inst.dm_file, experiment_dir)

    mod.setup_file_logging(experiment_dir)
    logging.info("instance-dir: %s", instance_dir)
    logging.info("experiment-dir: %s", experiment_dir)

    with (experiment_dir / "experiment_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)

    original_requests = _load_original_requests(inst.demand_file)
    if not original_requests:
        raise ValueError(f"No demand rows loaded from {inst.demand_file}")

    darp_requests = _original_requests_to_mod_darp_requests(original_requests)

    mod.write_darp_requests_csv(
        darp_requests,
        experiment_dir / "requests.csv",
        time_format="seconds",
    )
    mod.write_darp_vehicles_csv(
        darp_requests,
        experiment_dir / "vehicles.csv",
        capacity=cfg.darp_vehicle_capacity,
        time_format="seconds",
    )
    mod.write_darp_config_yaml(
        output_dir=experiment_dir,
        dm_filepath=dm_path_str,
        max_travel_time_delay_seconds=cfg.max_travel_time_delay_seconds,
        vehicle_capacity=cfg.darp_vehicle_capacity,
        darp_method=cfg.darp_benchmark_method,
        darp_experiment_parameters=cfg.darp_benchmark_experiment_parameters,
    )

    summary = {
        "mode": "mod_only_darp",
        "n_requests": len(darp_requests),
        "instance_dir": str(instance_dir),
        "demand_file": str(inst.demand_file),
        "experiment_dir": str(experiment_dir),
        "dm_filepath_in_darp_config": dm_path_str,
        "darp_benchmark_method": cfg.darp_benchmark_method,
        "darp_benchmark_experiment_parameters": cfg.darp_benchmark_experiment_parameters,
        "darp_vehicle_capacity": cfg.darp_vehicle_capacity,
        "max_travel_time_delay_seconds": cfg.max_travel_time_delay_seconds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (experiment_dir / "mod_only_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    experiment_config_path = experiment_dir / "experiment_ih.yaml"
    ok = darpbenchmark.experiments.run_experiment_using_config(
        str(experiment_config_path),
        executable_path=cfg.darp_benchmark_executable,
    )
    if not ok:
        logging.error("DARP-benchmark reported failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
