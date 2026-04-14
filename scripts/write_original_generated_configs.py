"""
Write self-contained instance directories + per-experiment directories.

Instances (copied demand + lines from repo ``test_data/``, local config.yaml):
  C:\\...\\Line Planning\\Instances\\original-generated\\<instance_stem>\\

Experiments (one folder each, ``experiment.yaml`` with ``mass_transport`` and ``solver``; ``results_dir`` omitted so outputs stay in that folder):
  C:\\...\\Line Planning\\Results\\original-generated\\<experiment_stem>\\

If the Google Drive tree cannot be created, the same structure is written under
``<repo>/experiments/original-generated/`` with ``Instances/`` and ``Results/``
siblings there.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "test_data"

LINE_PLANNING = Path(
    r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning"
)
GOOGLE_INSTANCES = LINE_PLANNING / "Instances" / "original-generated"
GOOGLE_EXPERIMENTS = LINE_PLANNING / "Results" / "original-generated"
# Shared distance matrix for generated ``original-generated`` instances (sibling of ``original-generated``).
CANONICAL_DM_H5 = LINE_PLANNING / "Instances" / "original" / "dm.h5"

REPO_BUNDLE_ROOT = REPO_ROOT / "experiments" / "original-generated"
REPO_INSTANCES = REPO_BUNDLE_ROOT / "Instances"
REPO_EXPERIMENTS = REPO_BUNDLE_ROOT / "Results"

DEFAULT_MASS_TRANSPORT: Dict[str, Any] = {
    "capacity": 30,
    "maximum_detour": 3,
    "cost_coefficient": 1,
    "max_frequency": 1,
}

DEFAULT_SOLVER: Dict[str, Any] = {
    "method": "ilp",
    "time_limit": 86400,
}

# (directory name under original-generated, demand filename, lines filename)
INSTANCE_SPECS: List[Tuple[str, str, str]] = [
    ("april_fhv_100", "OD_matrix_april_fhv.txt", "all_lines_nodes_1000_c5.txt"),
    ("april_fhv_march", "OD_matrix_march_fhv.txt", "all_lines_nodes_1000_c5.txt"),
    ("april_fhv_feb", "OD_matrix_feb_fhv.txt", "all_lines_nodes_1000_c5.txt"),
    ("april_fhv_50_percent", "OD_matrix_april_fhv_50_percent.txt", "all_lines_nodes_500_c5.txt"),
    ("april_fhv_10_percent", "OD_matrix_april_fhv_10_percent.txt", "all_lines_nodes_100_c5.txt"),
    ("april_fhv_1_percent", "OD_matrix_april_fhv_1_percent.txt", "all_lines_nodes_10_c5.txt"),
]

def _dm_filepath_relative_to_instance_dir(inst_dir: Path, dm_file: Path) -> str:
    """Relative path from ``inst_dir`` to ``dm_file`` (or absolute if on another drive)."""
    try:
        return Path(os.path.relpath(dm_file.resolve(), start=inst_dir.resolve())).as_posix()
    except ValueError:
        return str(dm_file.resolve())


def _dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def write_instance_bundle(
    instances_root: Path,
    stem: str,
    demand_name: str,
    lines_name: str,
    dm_file: Path = CANONICAL_DM_H5,
) -> Path:
    """
    Create ``instances_root/<stem>/`` with only the demand and candidate-lines
    files referenced by ``config.yaml`` (copied from repo ``test_data/``), plus
    ``config.yaml`` itself. ``dm_filepath`` is written relative to this folder
    (typically ``../../original/dm.h5`` under ``Instances/original-generated/``).
    Returns path to config.yaml.
    """
    inst_dir = instances_root / stem
    if inst_dir.exists():
        shutil.rmtree(inst_dir)
    inst_dir.mkdir(parents=True, exist_ok=True)

    if not TEST_DATA.is_dir():
        raise FileNotFoundError(f"Expected test_data directory at {TEST_DATA}")
    for name in (demand_name, lines_name):
        src = TEST_DATA / name
        if not src.is_file():
            raise FileNotFoundError(f"Expected instance input file at {src}")
        shutil.copy2(src, inst_dir / name)

    config = {
        "demand": {"filepath": f"./{demand_name}"},
        "lines": f"./{lines_name}",
        "dm_filepath": _dm_filepath_relative_to_instance_dir(inst_dir, dm_file),
    }
    config_path = inst_dir / "config.yaml"
    _dump(config_path, config)
    return config_path


def _experiment_yaml(
    *,
    instance_relpath: str,
    mass_transport: Optional[Dict[str, Any]] = None,
    solver: Optional[Dict[str, Any]] = None,
    budget: Any = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "instance": instance_relpath,
        "mass_transport": dict(
            DEFAULT_MASS_TRANSPORT if mass_transport is None else {**DEFAULT_MASS_TRANSPORT, **mass_transport}
        ),
        "solver": dict(DEFAULT_SOLVER if solver is None else {**DEFAULT_SOLVER, **solver}),
    }
    if budget is not None:
        out["budget"] = budget
    return out


def _clean_stale_flat_results_layout(results_original_generated: Path) -> None:
    """Remove obsolete flat layout under results (root *.yaml and ``instances/`` subfolder)."""
    if not results_original_generated.is_dir():
        return
    stale_inst = results_original_generated / "instances"
    if stale_inst.is_dir():
        shutil.rmtree(stale_inst, ignore_errors=True)
    for p in results_original_generated.glob("*.yaml"):
        try:
            p.unlink()
        except OSError:
            pass


def write_bundle(instances_root: Path, experiments_root: Path) -> None:
    instance_config_paths: Dict[str, Path] = {}
    for stem, demand, lines in INSTANCE_SPECS:
        instance_config_paths[stem] = write_instance_bundle(instances_root, stem, demand, lines)

    experiments: List[Tuple[str, str, Dict[str, Any]]] = [
        (
            "exp_test_original_current_MIP",
            "april_fhv_100",
            _experiment_yaml(instance_relpath=""),  # filled below
        ),
        ("exp_test_one_percent", "april_fhv_1_percent", _experiment_yaml(instance_relpath="")),
        ("exp_march_demand", "april_fhv_march", _experiment_yaml(instance_relpath="")),
        ("exp_feb_demand", "april_fhv_feb", _experiment_yaml(instance_relpath="")),
        ("exp_50_percent_demand", "april_fhv_50_percent", _experiment_yaml(instance_relpath="")),
        ("exp_10_percent_demand", "april_fhv_10_percent", _experiment_yaml(instance_relpath="")),
        (
            "exp_budget_30_000",
            "april_fhv_100",
            _experiment_yaml(instance_relpath="", budget=30_000),
        ),
        (
            "exp_budget_200_000",
            "april_fhv_100",
            _experiment_yaml(instance_relpath="", budget=200_000),
        ),
        (
            "exp_budget_500_000",
            "april_fhv_100",
            _experiment_yaml(instance_relpath="", budget=500_000),
        ),
        (
            "exp_stage_1_mod_costs",
            "april_fhv_100",
            _experiment_yaml(
                instance_relpath="",
                solver={**DEFAULT_SOLVER, "method": "ilp_with_mod_costs"},
            ),
        ),
        (
            "exp_stage_2_empty_trips",
            "april_fhv_100",
            _experiment_yaml(
                instance_relpath="",
                solver={**DEFAULT_SOLVER, "method": "ilp_with_empty_trips"},
            ),
        ),
        ("exp_unlimited_budget", "april_fhv_100", _experiment_yaml(instance_relpath="")),
    ]

    for exp_name, inst_stem, payload in experiments:
        exp_dir = experiments_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        inst_cfg = instance_config_paths[inst_stem]
        rel = os.path.relpath(inst_cfg, exp_dir).replace("\\", "/")
        payload["instance"] = rel
        _dump(exp_dir / "experiment.yaml", payload)


def main() -> int:
    try:
        GOOGLE_INSTANCES.mkdir(parents=True, exist_ok=True)
        GOOGLE_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
        _clean_stale_flat_results_layout(GOOGLE_EXPERIMENTS)
        write_bundle(GOOGLE_INSTANCES, GOOGLE_EXPERIMENTS)
        print(f"Wrote instances under {GOOGLE_INSTANCES}")
        print(f"Wrote experiments under {GOOGLE_EXPERIMENTS}")
    except OSError as exc:
        print(f"{GOOGLE_INSTANCES} / {GOOGLE_EXPERIMENTS}: {exc}", file=sys.stderr)
        REPO_INSTANCES.mkdir(parents=True, exist_ok=True)
        REPO_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
        _clean_stale_flat_results_layout(REPO_EXPERIMENTS)
        write_bundle(REPO_INSTANCES, REPO_EXPERIMENTS)
        print(f"Wrote bundle under {REPO_BUNDLE_ROOT}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
