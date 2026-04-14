"""
Load line-planning instance paths from a YAML file compatible with the
`Ridesharing_DARP_instances` layout (demand.filepath, dm_filepath, area_dir),
plus a required ``lines`` key for the candidate-lines file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union

import yaml


@dataclass(frozen=True)
class LinePlanningInstancePaths:
    """Resolved filesystem paths for one line-planning instance."""

    config_path: Path
    demand_file: Path
    lines_file: Path
    dm_file: Path


def _resolve_path(base_dir: Path, value: Union[str, Path]) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


def load_line_planning_instance_config(config_path: Path) -> LinePlanningInstancePaths:
    """
    Parse ``config_path`` (YAML) and return resolved paths.

    Required / expected keys:
    - ``demand`` mapping with ``filepath`` (DARP style), or a string path as ``demand``.
    - ``lines``: path to the candidate-lines text file.
    - ``dm_filepath`` and/or ``area_dir`` so a distance matrix can be resolved.
    """
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Instance config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, MutableMapping):
        raise ValueError(f"Instance config must be a YAML mapping, got {type(raw).__name__}")

    data: Dict[str, Any] = dict(raw)
    base_dir = config_path.parent

    lines_val = data.get("lines")
    if not lines_val:
        raise ValueError(
            f"{config_path}: missing required top-level key 'lines' (path to candidate lines file)."
        )
    lines_file = _resolve_path(base_dir, str(lines_val))

    demand_block = data.get("demand")
    if isinstance(demand_block, str) and demand_block.strip():
        demand_file = _resolve_path(base_dir, demand_block)
    elif isinstance(demand_block, Mapping):
        fp = demand_block.get("filepath")
        if not fp:
            raise ValueError(
                f"{config_path}: 'demand' is a mapping but has no 'filepath' "
                "(see Ridesharing_DARP_instances instance config)."
            )
        demand_file = _resolve_path(base_dir, str(fp))
    else:
        raise ValueError(
            f"{config_path}: expected 'demand' to be a string path or a mapping with 'filepath'."
        )

    dm_val = data.get("dm_filepath")
    dm_file: Optional[Path] = None
    if isinstance(dm_val, str) and dm_val.strip():
        dm_file = _resolve_path(base_dir, dm_val)

    if dm_file is None:
        area_val = data.get("area_dir")
        if isinstance(area_val, str) and area_val.strip():
            area_root = _resolve_path(base_dir, area_val)
            dm_file = _first_existing(
                [
                    area_root / "dm.h5",
                    area_root / "dm.hd5",
                    area_root / "dm.hdf5",
                    area_root / "dm.csv",
                ]
            )
            if dm_file is None:
                dm_file = area_root / "dm.h5"

    if dm_file is None:
        raise ValueError(
            f"{config_path}: set 'dm_filepath' and/or 'area_dir' so the distance matrix can be located."
        )

    return LinePlanningInstancePaths(
        config_path=config_path,
        demand_file=demand_file,
        lines_file=lines_file,
        dm_file=dm_file,
    )


def load_experiment_yaml(experiment_path: Path) -> Dict[str, Any]:
    experiment_path = experiment_path.resolve()
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    with experiment_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, MutableMapping):
        raise ValueError(
            f"Experiment config must be a YAML mapping, got {type(raw).__name__}"
        )
    return dict(raw)


def resolve_instance_config_path(experiment_path: Path, experiment: Mapping[str, Any]) -> Path:
    inst = experiment.get("instance")
    if not inst:
        raise ValueError(
            f"{experiment_path}: missing required key 'instance' (path to instance config.yaml)."
        )
    inst_path = Path(str(inst))
    if inst_path.is_absolute():
        return inst_path.resolve()
    return (experiment_path.parent / inst_path).resolve()


def resolve_results_dir(experiment_path: Path, experiment: Mapping[str, Any]) -> Path:
    """
    Resolve the directory where experiment outputs (logs, exports, metrics) are written.

    If ``results_dir`` is absent or empty, returns the directory containing the
    experiment YAML file.
    """
    experiment_path = experiment_path.resolve()
    rd = experiment.get("results_dir")
    if rd is None or (isinstance(rd, str) and not rd.strip()):
        return experiment_path.parent.resolve()
    p = Path(str(rd).strip())
    if p.is_absolute():
        return p.resolve()
    return (experiment_path.parent / p).resolve()
