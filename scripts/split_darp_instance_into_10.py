"""
Split a DARP instance's demand (requests.csv) into N smaller instances.

Requirements from user:
- Each request row is stored in exactly one smaller instance, unchanged.
- Output instances are written under a target directory.
- The distance matrix is not copied; new configs store a relative path to the original dm file.
- Vehicles can be skipped (not written into the new configs).

This script is intentionally conservative:
- It preserves *all* columns in requests.csv (including extra metadata like original_request_id).
- It keeps existing config keys (if config.yaml exists) and only updates demand.filepath and dm_filepath.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _read_requests_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _write_requests_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Write values exactly as they were read (strings), preserving column set/order.
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_yaml_if_possible(path: Path) -> Dict:
    """
    Load YAML if PyYAML is available; otherwise raise ImportError.
    """
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top-level in {path}, got {type(data).__name__}.")
    return data


def _dump_yaml_if_possible(path: Path, data: Dict) -> None:
    import yaml  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _partition_indices(n_items: int, n_parts: int) -> List[List[int]]:
    """
    Split range(n_items) into n_parts chunks with sizes differing by at most 1.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be > 0")
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    base = n_items // n_parts
    rem = n_items % n_parts
    parts: List[List[int]] = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        parts.append(list(range(start, start + size)))
        start += size
    return parts


def _ensure_relpath(target_dir: Path, original_path: Path) -> str:
    """
    Return a relative path from target_dir to original_path, using forward slashes
    for YAML portability.
    """
    rel = os.path.relpath(str(original_path), start=str(target_dir))
    return rel.replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a DARP instance demand into N smaller instances."
    )
    parser.add_argument(
        "--source-instance-dir",
        required=True,
        type=Path,
        help=r'Path to source instance folder (e.g. ...\instances\start_18-00\duration_02_h\max_delay_05_min).',
    )
    parser.add_argument(
        "--output-root-dir",
        required=True,
        type=Path,
        help=r"Directory where split instances will be created.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of smaller instances to create (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for shuffling before splitting (default: 0).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle requests; split in original file order.",
    )
    parser.add_argument(
        "--output-name-prefix",
        type=str,
        default="instance_",
        help="Prefix for created instance folders (default: instance_).",
    )
    args = parser.parse_args()

    source_dir: Path = args.source_instance_dir
    output_root: Path = args.output_root_dir
    n_parts: int = args.n

    requests_path = source_dir / "requests.csv"
    if not requests_path.exists():
        raise FileNotFoundError(f"Expected {requests_path} to exist.")

    config_path = source_dir / "config.yaml"
    config_exists = config_path.exists()

    fieldnames, rows = _read_requests_csv(requests_path)
    n_items = len(rows)

    order = list(range(n_items))
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(order)

    partitions = _partition_indices(n_items, n_parts)

    # Map partitions from 0..n_items-1 indices into shuffled order indices.
    partitions = [[order[i] for i in part] for part in partitions]

    # Load existing config if present.
    config_data: Dict = {}
    dm_original: Path | None = None
    area_root: Path | None = None
    if config_exists:
        try:
            config_data = _load_yaml_if_possible(config_path)
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to read/write config.yaml. "
                "Install with `pip install pyyaml` and rerun."
            ) from exc

        dm_val = config_data.get("dm_filepath")
        if isinstance(dm_val, str) and dm_val.strip():
            dm_original = Path(dm_val)

        # Resolve area_dir relative to the original instance directory.
        area_dir_val = config_data.get("area_dir")
        if isinstance(area_dir_val, str) and area_dir_val.strip():
            area_root = (source_dir / area_dir_val).resolve()

    # If config.yaml doesn't exist (or doesn't specify dm), infer dm.h5 from area root.
    if dm_original is None:
        # Common DARP structure: <area_root>/instances/<...>/config.yaml and <area_root>/dm.h5
        # source_dir is .../instances/start_.../duration_.../max_delay_... so parents[3] is <area_root>
        try:
            inferred_area_root = source_dir.parents[3]
        except IndexError:
            inferred_area_root = source_dir.parent
        dm_candidates = [inferred_area_root / "dm.h5", inferred_area_root / "dm.hdf5", inferred_area_root / "dm.csv"]
        dm_original = next((p for p in dm_candidates if p.exists()), dm_candidates[0])
        # If area_root was not resolved from config, fall back to inferred root.
        if area_root is None:
            area_root = inferred_area_root

    output_root.mkdir(parents=True, exist_ok=True)

    for part_idx, idxs in enumerate(partitions, start=1):
        # Keep stable output folder names.
        folder_name = f"{args.output_name_prefix}{part_idx:02d}"
        out_dir = output_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write subset requests.csv (rows unchanged).
        subset_rows = [rows[i] for i in idxs]
        _write_requests_csv(out_dir / "requests.csv", fieldnames, subset_rows)

        # Write config.yaml (copy + patch).
        # - demand.filepath must point to local requests.csv
        # - dm_filepath must be relative to original matrix file
        # - area_dir must be relative to original area directory (parent of map dir)
        # - vehicles can be omitted
        out_config: Dict = dict(config_data) if config_data else {}
        out_config["demand"] = dict(out_config.get("demand") or {})
        out_config["demand"]["filepath"] = "./requests.csv"
        out_config["dm_filepath"] = _ensure_relpath(out_dir, dm_original)
        if area_root is not None:
            out_config["area_dir"] = _ensure_relpath(out_dir, area_root)
        out_config.pop("vehicles", None)

        try:
            _dump_yaml_if_possible(out_dir / "config.yaml", out_config)
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to write config.yaml. "
                "Install with `pip install pyyaml` and rerun."
            ) from exc

    print(
        f"Wrote {n_parts} instances under {output_root} from {requests_path} "
        f"({n_items} requests total)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

