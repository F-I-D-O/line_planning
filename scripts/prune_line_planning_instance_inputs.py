"""
Remove files under a line-planning instance directory that are not referenced
by ``config.yaml`` (same path keys as ``lineplanning.instance_config``).

Only deletes regular files directly under the instance directory (not
subdirectories), so accidental extra folders are left untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Set

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lineplanning.instance_config import load_line_planning_instance_config


def _config_path_in_dir(instance_dir: Path) -> Path | None:
    for name in ("config.yaml", "config.yml", "instance_config.yaml", "instance_config.yml"):
        p = instance_dir / name
        if p.is_file():
            return p
    return None


def _collect_kept_resolved_files(config_path: Path) -> Set[Path]:
    inst_dir = config_path.parent.resolve()
    kept: Set[Path] = {config_path.resolve()}
    resolved = load_line_planning_instance_config(config_path)
    for p in (resolved.demand_file, resolved.lines_file, resolved.dm_file):
        rp = p.resolve()
        if rp.is_relative_to(inst_dir):
            kept.add(rp)

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if isinstance(raw, dict):
        area_val = raw.get("area_dir")
        if isinstance(area_val, str) and area_val.strip():
            area_root = (inst_dir / area_val).resolve()
            if area_root.is_relative_to(inst_dir) and area_root.is_dir():
                for fpath in area_root.rglob("*"):
                    if fpath.is_file():
                        kept.add(fpath.resolve())
    return kept


def prune_instance_directory(instance_dir: Path, *, dry_run: bool = False) -> list[Path]:
    """
    Delete top-level files under ``instance_dir`` that are not in the kept set.
    Returns paths that were removed (or would be removed if ``dry_run``).
    """
    instance_dir = instance_dir.resolve()
    cfg = _config_path_in_dir(instance_dir)
    if cfg is None:
        raise FileNotFoundError(f"No config.yaml (or alternate) under {instance_dir}")

    kept = _collect_kept_resolved_files(cfg)
    removed: list[Path] = []
    for entry in instance_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.resolve() in kept:
            continue
        removed.append(entry)
        if not dry_run:
            entry.unlink()
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Instance directories (each should contain config.yaml).",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print files that would be deleted without deleting.",
    )
    args = parser.parse_args()

    exit_code = 0
    for d in args.directories:
        d = d.expanduser()
        if not d.is_dir():
            print(f"Skip (not a directory): {d}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            removed = prune_instance_directory(d, dry_run=args.dry_run)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{d}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        label = "Would remove" if args.dry_run else "Removed"
        for p in removed:
            print(f"{label}: {p}")
        if not removed:
            print(f"{d}: nothing to prune")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
