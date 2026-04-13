"""
Write self-contained instance directories + per-experiment directories.

Instances (copied demand + lines from repo ``test_data/``, local config.yaml):
  C:\\...\\Line Planning\\Instances\\original-generated\\<instance_stem>\\

Experiments (one folder each, ``experiment.yaml`` + results_dir ``.``):
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

REPO_BUNDLE_ROOT = REPO_ROOT / "experiments" / "original-generated"
REPO_INSTANCES = REPO_BUNDLE_ROOT / "Instances"
REPO_EXPERIMENTS = REPO_BUNDLE_ROOT / "Results"

DEFAULT_LINE_INSTANCE: Dict[str, Any] = {
    "cost": 1,
    "max_length": 15,
    "min_length": 8,
    "proba": 0.1,
    "capacity": 30,
    "detour_factor": 3,
    "method": 3,
    "granularity": 1,
}

DEFAULT_SOLVER: Dict[str, Any] = {
    "use_model_with_mod_costs": False,
    "use_model_with_empty_trips": False,
    "run_proposed_method": False,
    "allowed_time": 86400,
    "fixed_cost": 1,
    "max_frequency": 1,
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

# dm: shared area matrix next to ``original-generated`` (not copied — large file)
DM_REL_TO_INSTANCE = "../original/dm.h5"


def _dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def write_instance_bundle(instances_root: Path, stem: str, demand_name: str, lines_name: str) -> Path:
    """
    Create ``instances_root/<stem>/`` with a full copy of repo ``test_data/`` files
    (so the bundle does not reference the repository) plus ``config.yaml``.
    Returns path to config.yaml.
    """
    inst_dir = instances_root / stem
    if inst_dir.exists():
        shutil.rmtree(inst_dir)
    inst_dir.mkdir(parents=True, exist_ok=True)

    if not TEST_DATA.is_dir():
        raise FileNotFoundError(f"Expected test_data directory at {TEST_DATA}")
    for f in sorted(TEST_DATA.iterdir()):
        if f.is_file():
            shutil.copy2(f, inst_dir / f.name)

    config = {
        "demand": {"filepath": f"./{demand_name}"},
        "lines": f"./{lines_name}",
        "dm_filepath": DM_REL_TO_INSTANCE,
    }
    config_path = inst_dir / "config.yaml"
    _dump(config_path, config)
    return config_path


def _experiment_yaml(
    *,
    instance_relpath: str,
    line_instance: Optional[Dict[str, Any]] = None,
    solver: Optional[Dict[str, Any]] = None,
    budget: Any = None,
    random_seed: int = 127,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "instance": instance_relpath,
        "results_dir": ".",
        "random_seed": random_seed,
        "line_instance": dict(
            DEFAULT_LINE_INSTANCE if line_instance is None else {**DEFAULT_LINE_INSTANCE, **line_instance}
        ),
        "solver": dict(DEFAULT_SOLVER if solver is None else {**DEFAULT_SOLVER, **solver}),
    }
    if budget is not None:
        out["budget"] = budget
    return out


def _clean_legacy_google_results_flat_layout(results_original_generated: Path) -> None:
    """Remove previous single-folder layout (root *.yaml and ``instances/``)."""
    if not results_original_generated.is_dir():
        return
    legacy_inst = results_original_generated / "instances"
    if legacy_inst.is_dir():
        shutil.rmtree(legacy_inst, ignore_errors=True)
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
                solver={**DEFAULT_SOLVER, "use_model_with_mod_costs": True},
            ),
        ),
        (
            "exp_stage_2_empty_trips",
            "april_fhv_100",
            _experiment_yaml(
                instance_relpath="",
                solver={**DEFAULT_SOLVER, "use_model_with_empty_trips": True},
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
        _clean_legacy_google_results_flat_layout(GOOGLE_EXPERIMENTS)
        write_bundle(GOOGLE_INSTANCES, GOOGLE_EXPERIMENTS)
        print(f"Wrote instances under {GOOGLE_INSTANCES}")
        print(f"Wrote experiments under {GOOGLE_EXPERIMENTS}")
    except OSError as exc:
        print(f"{GOOGLE_INSTANCES} / {GOOGLE_EXPERIMENTS}: {exc}", file=sys.stderr)
        REPO_INSTANCES.mkdir(parents=True, exist_ok=True)
        REPO_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
        _clean_legacy_google_results_flat_layout(REPO_EXPERIMENTS)
        write_bundle(REPO_INSTANCES, REPO_EXPERIMENTS)
        print(f"Wrote bundle under {REPO_BUNDLE_ROOT}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
