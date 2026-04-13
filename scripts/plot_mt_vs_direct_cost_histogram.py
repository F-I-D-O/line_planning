"""Jupyter-style script: Run Cell on each `# %%` region (VS Code / Cursor).

Loads a ``line_instance`` (``lineplanning.instance``), which reads preprocessing
from ``<results_dir>/preprocessing/*.csv`` when the cache exists (same as a normal
solve). For each request, compares total travel time of the **best** mass-transit
option (min over candidate lines of first-mile + on-line + last-mile) to the
**non-MT** option (direct O--D time stored on the synthetic last ``TripOption``).

Histogram: x = percent cost difference ``(best_MT - direct) / direct * 100`` (negative
means MT is faster), y = number of requests; bins are 10 percentage points wide.

Requires: plotly, pandas, numpy, pyyaml, darpinstances (for dm), tqdm.
"""

from __future__ import annotations

# %% Configuration — edit this cell only
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml

import lineplanning.instance
import lineplanning.log  # noqa: F401 — configures logging

# Instance folder (must contain config.yaml / config.yml with demand + dm paths)
INSTANCE_DIR = Path(
    r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning\Instances\manhattan-2_h-10_percent\instance_01"
)
CANDIDATE_LINES_FILE = INSTANCE_DIR / "lines.txt"

# Must match the preprocessing cache (same as line_planning / MoD-aware scripts)
RESULTS_DIR = Path(
    r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning\Results\manhattan-2_h-10_percent\instance_01\mod-aware"
)
MAXIMUM_DETOUR = 3
GRANULARITY = 1

# If set, write interactive HTML; if None, skip
OUTPUT_HTML: Path | None = None
SHOW_IN_BROWSER = True
FIGURE_TITLE: str | None = None

# %% Resolve demand / dm from config (same pattern as MoD-aware_line_selection.py)


def _resolve_config_path(instance_dir_path: Path) -> Path:
    for name in ("config.yaml", "config.yml", "instance_config.yaml", "instance_config.yml"):
        p = instance_dir_path / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find instance config. Tried config.yaml, config.yml, "
        "instance_config.yaml, instance_config.yml under "
        f"{instance_dir_path}"
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


# %% Build instance (loads preprocessing cache under RESULTS_DIR / preprocessing)


demand_file, dm_file = _load_demand_and_dm_from_instance_config(INSTANCE_DIR)

line_inst = lineplanning.instance.line_instance(
    candidate_lines_file=CANDIDATE_LINES_FILE,
    capacity=30,
    maximum_detour=MAXIMUM_DETOUR,
    granularity=GRANULARITY,
    demand_file=demand_file,
    results_dir=RESULTS_DIR,
    dm_file=dm_file,
)

# %% Compute percent differences (best MT vs direct)


def _mt_total_time(opt: lineplanning.instance.TripOption) -> float:
    return float(opt.first_mile_cost + opt.last_mile_cost + opt.mt_cost)


def _is_valid_mt_option(opt: lineplanning.instance.TripOption) -> bool:
    return opt.mt_pickup_node != -1


percent_diffs: list[float] = []
skipped_no_direct = 0
skipped_no_mt = 0

for p in range(line_inst.nb_pass):
    opts = line_inst.optimal_trip_options[p]
    if len(opts) < 2:
        raise ValueError(f"Passenger {p}: expected at least one MT row and one no-MT row")

    direct = float(opts[-1].first_mile_cost)
    if direct <= 0:
        skipped_no_direct += 1
        continue

    mt_rows = opts[:-1]
    valid = [o for o in mt_rows if _is_valid_mt_option(o)]
    if not valid:
        skipped_no_mt += 1
        continue

    # best_mt = min(_mt_total_time(o) for o in valid)
    best_mt = min(opt.first_mile_cost + opt.last_mile_cost for opt in valid)
    pct = (best_mt - direct) / direct * 100.0
    percent_diffs.append(pct)

logging.info(
    "Requests: %d with histogram, %d no valid MT option, %d zero direct time",
    len(percent_diffs),
    skipped_no_mt,
    skipped_no_direct,
)

# %% Plotly histogram (10% bin width)


if not percent_diffs:
    raise RuntimeError("No data to plot (empty percent_diffs).")

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

title = FIGURE_TITLE
if title is None:
    title = (
        f"Best MT vs direct travel time — {INSTANCE_DIR.name}<br>"
        "<sub>x = (best_MT_time − direct_OD_time) / direct_OD_time × 100 "
        "(negative ⇒ MT faster)</sub>"
    )

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

if OUTPUT_HTML is not None:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    logging.info("Wrote %s", OUTPUT_HTML)

if SHOW_IN_BROWSER:
    fig.show()

# %%
