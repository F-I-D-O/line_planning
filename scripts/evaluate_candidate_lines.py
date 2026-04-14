"""
Evaluate candidate line sets: candidate lines, trip-option preprocessing, metrics row,
and histogram under a single run folder.

**Directories:**
- *Instance dir* (--instance-dir): config YAML; demand and DM paths from config.
- *Output dir* (--output-dir): parent folder; only ``metrics.csv`` at the root.
  Each run uses a subfolder ``v_<N>-<sanitized_label>/``. If the **latest** version index
  (highest ``N`` among ``v_*`` children) already has this label, that folder is **reused**;
  otherwise ``N`` is one more than the max of folder indices and ``metrics.csv`` hints.
  Subfolders contain ``lines.txt``,
  optional ``candidate_lines.gpkg``, histogram HTML, and ``.line_eval_complete`` when finished.
  Trip-option preprocessing CSV caches live under **instance dir** ``preprocessing/`` (shared
  across runs when demand, lines path, and detour match).

**Idempotent behavior (no extra flags):**
- If ``.line_eval_complete`` exists (or legacy: histogram plus preprocessing CSV for this run
  under the instance ``preprocessing/`` folder, or old ``trip_options_preprocessing.csv``), exit without doing work.
- If ``lines.txt`` exists, skip candidate line generation.
- If a valid preprocessing CSV already exists for this demand / lines path / maximum detour,
  ``line_instance`` loads it (no recomputation).

**Metrics "best" MT option:** minimizes ``first_mile + last_mile`` (ignores ``mt_cost``).

**Histogram:** same criterion — best MT by minimum ``first_mile + last_mile`` vs direct OD time
in ``opts[-1].first_mile_cost``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml

import lineplanning.log  # noqa: F401
from lineplanning.candidate_lines import (
    DEFAULT_DETOUR_SKELETON,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_LENGTH,
    DEFAULT_MIN_START_END_DISTANCE,
    generate_candidate_lines,
)
from lineplanning.instance import TripOption, line_instance, preprocessing_csv_path

_EVAL_COMPLETE = ".line_eval_complete"


def _resolve_config_path(instance_dir_path: Path) -> Path:
    for name in ("config.yaml", "config.yml", "instance_config.yaml", "instance_config.yml"):
        p = instance_dir_path / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find instance config. Tried config.yaml, config.yml, "
        f"instance_config.yaml, instance_config.yml under {instance_dir_path}"
    )


def _coerce_path(v: object) -> Path:
    if isinstance(v, Path):
        return v
    if isinstance(v, str):
        return Path(v)
    raise TypeError(f"Expected str or Path, got {type(v).__name__}")


def _load_demand_and_dm_from_instance_config(instance_dir_path: Path) -> Tuple[Path, Path]:
    config_path = _resolve_config_path(instance_dir_path)
    config_dir = config_path.parent
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(cfg).__name__}")

    demand_cfg = cfg.get("demand")
    dm_filepath = cfg.get("dm_filepath")
    if not isinstance(demand_cfg, dict):
        raise ValueError(f"Expected config['demand'] to be a mapping in {config_path}")
    demand_filepath = demand_cfg.get("filepath")
    if not demand_filepath:
        raise KeyError(f"Missing demand filepath in {config_path}")
    if not dm_filepath:
        raise KeyError(f"Missing dm_filepath in {config_path}")

    demand_path = _coerce_path(demand_filepath)
    dm_path = _coerce_path(dm_filepath)
    if not demand_path.is_absolute():
        demand_path = (config_dir / demand_path).resolve()
    if not dm_path.is_absolute():
        dm_path = (config_dir / dm_path).resolve()
    return demand_path, dm_path


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _resolve_area_path_from_config(instance_dir: Path) -> Path:
    try:
        config_path = _resolve_config_path(instance_dir)
    except FileNotFoundError:
        logging.info("No instance config; using instance directory as area path.")
        return instance_dir
    config = _load_yaml_mapping(config_path)
    raw = config.get("area_dir")
    if not isinstance(raw, str) or not raw.strip():
        logging.info("No area_dir in config; using instance directory as area path.")
        return instance_dir
    area_path = Path(raw)
    if not area_path.is_absolute():
        area_path = (instance_dir / area_path).resolve()
    return area_path


def _sanitize_label(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip())
    return s or "run"


def _max_version_index_from_v_dirs(work_root: Path) -> int:
    """Largest N among child dirs named ``v_N-...``; -1 if none."""
    best = -1
    if not work_root.is_dir():
        return -1
    for p in work_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^v_(\d+)-", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def _metrics_csv_version_hint(work_root: Path) -> int:
    """
    Max version index implied by ``metrics.csv``: max ``version`` column if present,
    else (number of data rows - 1). Returns -1 if no file or empty body.
    """
    path = work_root / "metrics.csv"
    if not path.exists():
        return -1
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                return -1
            rows = list(reader)
    except OSError as exc:
        logging.warning("Could not read %s for version hint: %s", path, exc)
        return -1
    if not rows:
        return -1
    if "version" in fieldnames:
        vals: list[int] = []
        for row in rows:
            v = row.get("version", "")
            if v is None or str(v).strip() == "":
                continue
            try:
                vals.append(int(float(str(v).strip())))
            except ValueError:
                continue
        return max(vals) if vals else -1
    return len(rows) - 1


def _next_version_index(work_root: Path) -> int:
    return max(_max_version_index_from_v_dirs(work_root), _metrics_csv_version_hint(work_root)) + 1


def _resolve_run_dir_and_version(work_root: Path, safe: str) -> Tuple[Path, int]:
    """
    If ``v_{max_n}-{safe}`` exists where ``max_n`` is the largest ``v_*`` index under
    ``work_root``, reuse it. Otherwise use ``v_{_next_version_index}-{safe}``.
    """
    max_n = _max_version_index_from_v_dirs(work_root)
    if max_n >= 0:
        reuse_dir = work_root / f"v_{max_n}-{safe}"
        if reuse_dir.is_dir():
            logging.info("Reusing run folder %s (latest version index matches this label).", reuse_dir)
            return reuse_dir, max_n
    version_index = _next_version_index(work_root)
    run_dir = work_root / f"v_{version_index}-{safe}"
    return run_dir, version_index


def _parse_version_from_run_dir_name(name: str) -> Optional[int]:
    m = re.match(r"^v_(\d+)-", name)
    return int(m.group(1)) if m else None


def _evaluation_complete(
    run_dir: Path,
    instance_dir: Path,
    demand_file: Path,
    maximum_detour: int,
) -> bool:
    if (run_dir / _EVAL_COMPLETE).is_file():
        return True
    hist = run_dir / "mt_vs_direct_histogram.html"
    if not hist.is_file():
        return False
    if (run_dir / "trip_options_preprocessing.csv").is_file():
        return True
    candidate_lines = run_dir / "lines.txt"
    cache_csv = preprocessing_csv_path(
        instance_dir / "preprocessing",
        demand_file,
        candidate_lines,
        maximum_detour,
    )
    return cache_csv.is_file()


def _is_valid_mt_option(opt: TripOption) -> bool:
    return opt.mt_pickup_node != -1


def _mod_cost(opt: TripOption) -> float:
    return float(opt.first_mile_cost + opt.last_mile_cost)


def _total_time(opt: TripOption) -> float:
    return float(opt.first_mile_cost + opt.last_mile_cost + opt.mt_cost)


def _compute_metrics_row(
    inst: line_instance,
    improvement_name: str,
    version: int,
    run_folder: str,
) -> Dict[str, Any]:
    rel_diffs_best: list[float] = []
    rel_diffs_top10: list[float] = []
    mod_costs_best: list[float] = []
    total_times_best: list[float] = []
    n_mod_worse = 0
    n_no_valid_mt = 0
    mean_direct_vals: list[float] = []

    for p in range(inst.nb_pass):
        opts = inst.optimal_trip_options[p]
        if len(opts) < 2:
            raise ValueError(f"Passenger {p}: expected MT rows and synthetic no-MT row")

        direct = float(opts[-1].first_mile_cost)

        mt_rows = opts[:-1]
        valid = [o for o in mt_rows if _is_valid_mt_option(o)]
        if not valid:
            n_no_valid_mt += 1
            continue

        best_mt = min(valid, key=_mod_cost)
        mod_b = _mod_cost(best_mt)
        tot_b = _total_time(best_mt)
        mod_costs_best.append(mod_b)
        total_times_best.append(tot_b)
        mean_direct_vals.append(direct)
        rel_diffs_best.append((mod_b - direct) / direct)

        if mod_b > direct:
            n_mod_worse += 1

        by_mod = sorted(valid, key=_mod_cost)
        top = by_mod[:10]
        avg_mod = float(np.mean([_mod_cost(o) for o in top]))
        rel_diffs_top10.append((avg_mod - direct) / direct)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row: Dict[str, Any] = {
        "version": version,
        "run_folder": run_folder,
        "improvement_name": improvement_name,
        "timestamp": ts,
        "n_used_metrics": len(rel_diffs_best),
        "sum_mod_cost_best": float(np.sum(mod_costs_best)) if mod_costs_best else "",
        "mean_rel_diff_best": float(np.mean(rel_diffs_best)) if rel_diffs_best else "",
        "mean_rel_diff_top10_mod": float(np.mean(rel_diffs_top10)) if rel_diffs_top10 else "",
        "n_mod_worse_than_direct": n_mod_worse,
        "median_rel_diff_best": float(np.median(rel_diffs_best)) if rel_diffs_best else "",
        "n_no_valid_mt": n_no_valid_mt,
        "mean_total_time_best": float(np.mean(total_times_best)) if total_times_best else "",
        "mean_direct_time": float(np.mean(mean_direct_vals)) if mean_direct_vals else "",
        "nb_lines": inst.nb_lines,
    }
    return row


_METRICS_COLUMNS = [
    "version",
    "run_folder",
    "improvement_name",
    "timestamp",
    "n_used_metrics",
    "sum_mod_cost_best",
    "mean_rel_diff_best",
    "mean_rel_diff_top10_mod",
    "n_mod_worse_than_direct",
    "median_rel_diff_best",
    "n_no_valid_mt",
    "mean_total_time_best",
    "mean_direct_time",
    "nb_lines",
]


def _append_metrics_csv(work_root: Path, row: Dict[str, Any]) -> None:
    path = work_root / "metrics.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_METRICS_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _METRICS_COLUMNS})


def _histogram_percent_diffs(inst: line_instance) -> list[float]:
    """Same selection as plot_mt_vs_direct_cost_histogram.py (min FM+LM among valid MT)."""
    percent_diffs: list[float] = []
    for p in range(inst.nb_pass):
        opts = inst.optimal_trip_options[p]
        if len(opts) < 2:
            raise ValueError(f"Passenger {p}: expected at least one MT row and one no-MT row")
        direct = float(opts[-1].first_mile_cost)
        mt_rows = opts[:-1]
        valid = [o for o in mt_rows if _is_valid_mt_option(o)]
        if not valid:
            continue
        best_mt = min(_mod_cost(o) for o in valid)
        percent_diffs.append((best_mt - direct) / direct * 100.0)
    return percent_diffs


def _histogram_figure(percent_diffs: list[float], title: str) -> go.Figure:
    x = np.asarray(percent_diffs, dtype=float)
    lo = np.floor(np.min(x) / 10.0) * 10.0
    hi = np.ceil(np.max(x) / 10.0) * 10.0
    if lo == hi:
        lo -= 10.0
        hi += 10.0
    bin_edges = np.arange(lo, hi + 10.0, 10.0)
    counts, edges = np.histogram(x, bins=bin_edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hover = [
        f"[{edges[i]:.0f}%, {edges[i + 1]:.0f}%) — {int(counts[i])} requests"
        for i in range(len(counts))
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=centers,
                y=counts,
                width=9.0,
                hovertext=hover,
                hoverinfo="text",
                marker_line_width=1,
                marker_line_color="white",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Percent cost difference (10% bins)",
        yaxis_title="Number of requests",
        bargap=0.05,
        template="plotly_white",
    )
    return fig


def _write_histogram_html(
    percent_diffs: list[float],
    output_path: Path,
    title: str,
) -> None:
    if not percent_diffs:
        logging.warning("No histogram data; skipping HTML export")
        return
    fig = _histogram_figure(percent_diffs, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    logging.info("Wrote histogram %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate lines under output-dir/v_<N>-<label>/; append metrics to output-dir/metrics.csv. "
            "Skips steps automatically when outputs already exist."
        )
    )
    parser.add_argument(
        "--instance-dir",
        type=Path,
        required=True,
        help="Input instance folder (config YAML; demand and DM paths resolved from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Parent folder: metrics.csv at root; run artifacts under v_<N>-<sanitized-improvement-name>/.",
    )
    parser.add_argument(
        "--improvement-name",
        "--label",
        dest="improvement_name",
        type=str,
        required=True,
        help="Label suffix in folder v_<N>-<sanitized_label> and in metrics.csv.",
    )
    parser.add_argument("--number-of-stops", type=int, required=True)
    parser.add_argument("--nb-lines", type=int, required=True)
    parser.add_argument("--maximum-detour", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=30)
    parser.add_argument("--min-length-line", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--max-length-line", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--min-start-end-distance",
        type=int,
        default=DEFAULT_MIN_START_END_DISTANCE,
    )
    parser.add_argument("--detour-skeleton", type=int, default=DEFAULT_DETOUR_SKELETON)
    parser.add_argument("--area-name", type=str, default=None)
    parser.add_argument(
        "--no-geopackage",
        action="store_true",
        help="Do not write candidate_lines.gpkg during generation.",
    )
    parser.add_argument("--geopackage-path", type=Path, default=None)
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Open the Plotly histogram in a browser.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, ...). Default: INFO.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    instance_dir = args.instance_dir.resolve()
    work_root = args.output_dir.resolve()
    demand_file, dm_file = _load_demand_and_dm_from_instance_config(instance_dir)
    safe = _sanitize_label(args.improvement_name)
    work_root.mkdir(parents=True, exist_ok=True)
    run_dir, version_index = _resolve_run_dir_and_version(work_root, safe)
    run_dir.mkdir(parents=True, exist_ok=True)

    if _evaluation_complete(run_dir, instance_dir, demand_file, args.maximum_detour):
        logging.info(
            "Evaluation already complete for %s; skipping (delete histogram, preprocessing "
            "CSV under preprocessing/, and %s to redo).",
            run_dir,
            _EVAL_COMPLETE,
        )
        marker = run_dir / _EVAL_COMPLETE
        if not marker.is_file():
            marker.write_text(
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ\n") + "legacy_detected\n",
                encoding="utf-8",
            )
        return

    candidate_lines_file = run_dir / "lines.txt"
    if not candidate_lines_file.is_file():
        logging.info("Generating candidate lines into %s", run_dir)
        area_path = _resolve_area_path_from_config(instance_dir)
        generate_candidate_lines(
            area_path=area_path,
            output_path=run_dir,
            number_of_stops=args.number_of_stops,
            nb_lines=args.nb_lines,
            min_length=args.min_length_line,
            max_length=args.max_length_line,
            min_start_end_distance=args.min_start_end_distance,
            detour_skeleton=args.detour_skeleton,
            area_name=args.area_name,
            export_geopackage=not args.no_geopackage,
            geopackage_path=args.geopackage_path,
        )
    else:
        logging.info("lines.txt exists at %s; skipping candidate line generation.", candidate_lines_file)

    cache_csv = preprocessing_csv_path(
        instance_dir / "preprocessing",
        demand_file,
        candidate_lines_file,
        args.maximum_detour,
    )
    if cache_csv.is_file():
        logging.info("Preprocessing cache present at %s; line_instance will load it if valid.", cache_csv)
    else:
        logging.info("No preprocessing cache yet; line_instance will compute preprocessing.")

    inst = line_instance(
        candidate_lines_file=candidate_lines_file,
        capacity=args.capacity,
        maximum_detour=args.maximum_detour,
        demand_file=demand_file,
        preprocessing_dir=instance_dir / "preprocessing",
        dm_file=dm_file,
    )

    if not cache_csv.exists():
        logging.warning("Preprocessing cache not found at %s after line_instance", cache_csv)

    parsed_v = _parse_version_from_run_dir_name(run_dir.name)
    version_for_row = parsed_v if parsed_v is not None else version_index
    row = _compute_metrics_row(
        inst, args.improvement_name, version_for_row, run_dir.name
    )
    _append_metrics_csv(work_root, row)

    pct = _histogram_percent_diffs(inst)
    hist_title = (
        f"Best MT vs direct travel time — {run_dir.name}<br>"
        "<sub>x = (best_FM+LM − direct_OD) / direct_OD × 100 (negative ⇒ less MoD time on MT option)</sub>"
    )
    _write_histogram_html(pct, run_dir / "mt_vs_direct_histogram.html", hist_title)

    (run_dir / _EVAL_COMPLETE).write_text(
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ\n"), encoding="utf-8"
    )
    logging.info("Wrote completion marker %s", run_dir / _EVAL_COMPLETE)

    if args.show_browser and pct:
        _histogram_figure(pct, hist_title).show()


if __name__ == "__main__":
    main()
