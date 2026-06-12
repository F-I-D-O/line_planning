"""Jupyter-style script: use Run Cell on each `# %%` region (VS Code / Cursor).

Data: ``RESULTS_DIR/iteration_<k>/metrics.json`` (ILP line_cost, mod_cost estimated),
``config.yaml-solution.json`` (+ CSVs) for real MoD, ``used_lines.csv`` for line counts,
and ``passenger_assignments.csv`` for MoD-only vs MT vs served passenger counts.

Optional **MoD-only** folder (default ``<instance>/MoD-only`` next to ``RESULTS_DIR``): standalone
DARP run with ``config.yaml-solution.json``; draws a horizontal **MoD-only DARP total** (root
``cost``) on the cost subplot.

Set environment variable ``PLOT_ITERATION_COSTS_DEBUG=1`` to log per-iteration cost fields, full
tracebacks on MoD-real failures, and a table of collected rows (no separate debugger connection).

Requires: plotly, pandas, darpinstances.
"""

from __future__ import annotations

# %% Configuration — edit this cell only
# %% Imports
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import darpinstances.inout
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

_PLOG = logging.getLogger(__name__)


def _plot_iteration_costs_debug() -> bool:
    return os.environ.get("PLOT_ITERATION_COSTS_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _ensure_plot_iteration_costs_debug_logging() -> None:
    if not _plot_iteration_costs_debug():
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")


# MoD-aware results folder containing iteration_1, iteration_2, ...
RESULTS_DIR = Path(
    r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning\Results\LODES"
)

# If set, write interactive HTML here; if None, no file is written
OUTPUT_HTML: Path | None = None

# If True, open the figure in the default browser (requires a Plotly renderer)
SHOW_IN_BROWSER = True

# Plot title; None uses the results folder name
FIGURE_TITLE: str | None = None

# MoD-only DARP baseline directory (e.g. .../instance_01/MoD-only). None = RESULTS_DIR.parent / "MoD-only"
MOD_ONLY_RESULTS_DIR: Path | None = None




# %% Helpers (from MoD-aware pipeline; cannot import MoD-aware_line_selection.py on load)


def _scalar_float(v: Any) -> float:
    """Python ``float`` for plot math; avoids ``TypeError`` on ``pd.NA`` / nullable pandas scalars."""
    if v is None:
        return float("nan")
    if pd.api.types.is_scalar(v) and pd.isna(v):
        return float("nan")
    return float(v)


def _load_darp_requests_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "id": "int64",
            "original_request_id": "int64",
            "origin": "int64",
            "destination": "int64",
        },
    )
    df["time"] = df["time"].astype("float64")
    return df


def _load_request_assignments_csv(path: Path) -> pd.DataFrame:
    """
    Same semantics as ``load_request_assignments_csv`` in ``MoD-aware_line_selection.py``:
    strip string ``line`` cells, map ``Dropped`` to rejected, integer route indices as ``line``.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig").sort_values("passenger").reset_index(drop=True)
    n = len(raw)
    kinds: List[str] = []
    line_indices: List[Optional[int]] = []
    for cell in raw["line"]:
        if pd.isna(cell):
            kinds.append("rejected")
            line_indices.append(None)
            continue
        if isinstance(cell, str):
            token = cell.strip()
        else:
            token = cell
        if token == "no_MT":
            kinds.append("no_MT")
            line_indices.append(None)
        elif token == "rejected":
            kinds.append("rejected")
            line_indices.append(None)
        elif str(token) == "Dropped":
            kinds.append("rejected")
            line_indices.append(None)
        else:
            kinds.append("line")
            line_indices.append(int(token))

    return pd.DataFrame(
        {
            "original_id": np.arange(n, dtype=np.int64),
            "kind": kinds,
            "line_idx": pd.array(line_indices, dtype="Int64"),
        }
    )


def _darp_id_to_line_planning_leg(
    darp_requests: pd.DataFrame,
    request_assignments: pd.DataFrame,
) -> Dict[int, str]:
    """Match ``_darp_id_to_line_planning_leg`` in MoD-aware_line_selection.py."""
    asg = request_assignments.set_index("original_id")
    by_original = (
        darp_requests.groupby("original_request_id", sort=False)["id"]
        .agg(lambda s: sorted(s.tolist()))
        .to_dict()
    )

    out: Dict[int, str] = {}
    for oid, ids_sorted in by_original.items():
        row = asg.loc[oid]
        kind = str(row["kind"])
        line_idx = row["line_idx"]
        assert kind != "rejected", f"DARP row(s) for original {oid} but assignment is rejected"
        if kind == "no_MT" or line_idx is None or pd.isna(line_idx):
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


def _compute_per_darp_request_costs(
    solution: dict,
    darp_requests: pd.DataFrame,
    request_assignments: pd.DataFrame,
) -> Dict[int, Tuple[float, float]]:
    """Match ``compute_per_darp_request_costs`` in MoD-aware_line_selection.py."""
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

    expected = {int(x) for x in darp_requests["id"].tolist()}
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


def _aggregate_mod_costs_for_original_requests(
    darp_request_leg_costs: Dict[int, Tuple[float, float]],
    darp_requests: pd.DataFrame,
    request_assignments: pd.DataFrame,
) -> dict:
    costs_df = pd.DataFrame(
        [(rid, fm, lm) for rid, (fm, lm) in darp_request_leg_costs.items()],
        columns=["id", "fm", "lm"],
    )
    costs_df["fm"] = costs_df["fm"].astype("float64")
    costs_df["lm"] = costs_df["lm"].astype("float64")
    # Match ``aggregate_mod_costs_for_original_requests`` in MoD-aware_line_selection.py:
    # pair first/last mile rows by ascending DARP ``id`` within each original.
    merged = darp_requests[["id", "original_request_id"]].merge(costs_df, on="id", how="inner")
    merged = merged.sort_values(["original_request_id", "id"], kind="mergesort")
    # Do not use ``groupby(...).nth(1)`` for the second leg: in some pandas versions the
    # ``nth(1)`` frame can omit group keys that ``size()`` still reports as size 2, which
    # then raises ``KeyError`` on ``.loc[oid]`` (e.g. KeyError: 0).
    merged["_leg_seq"] = merged.groupby("original_request_id", sort=False).cumcount()
    sizes = merged.groupby("original_request_id", sort=False).size()
    first_leg = merged.loc[merged["_leg_seq"] == 0].drop(columns=["_leg_seq"]).set_index(
        "original_request_id"
    )
    second_leg = merged.loc[merged["_leg_seq"] == 1].drop(columns=["_leg_seq"]).set_index(
        "original_request_id"
    )

    asg = request_assignments.set_index("original_id")
    original_request_costs: dict = {}

    for original_id in range(len(request_assignments)):
        row = asg.loc[original_id]
        kind = str(row["kind"])
        line_idx = row["line_idx"]
        if kind == "rejected":
            continue
        oid = int(original_id)
        n = int(sizes.loc[oid]) if oid in sizes.index else 0
        if kind == "no_MT" or line_idx is None or pd.isna(line_idx):
            assert n == 1, (
                f"MoD-only original request {oid} expected 1 DARP request, found {n}"
            )
            r = first_leg.loc[oid]
            original_request_costs[oid] = (_scalar_float(r["fm"]), _scalar_float(r["lm"]))
        else:
            assert n == 2, (
                f"Line-assigned original request {oid} expected 2 DARP requests, found {n}"
            )
            r0 = first_leg.loc[oid]
            r1 = second_leg.loc[oid]
            original_request_costs[oid] = (
                _scalar_float(r0["fm"]) + _scalar_float(r0["lm"]),
                _scalar_float(r1["fm"]) + _scalar_float(r1["lm"]),
            )

    return original_request_costs


def _sum_mod_real_from_darp_iteration(iter_dir: Path) -> Optional[float]:
    solution_path = iter_dir / "config.yaml-solution.json"
    requests_path = iter_dir / "requests.csv"
    assignments_path = iter_dir / "passenger_assignments.csv"
    solution = darpinstances.inout.load_json(solution_path)
    darp_requests = _load_darp_requests_csv(requests_path)
    request_assignments = _load_request_assignments_csv(assignments_path)
    darp_request_leg_costs = _compute_per_darp_request_costs(
        solution, darp_requests, request_assignments
    )
    agg = _aggregate_mod_costs_for_original_requests(
        darp_request_leg_costs, darp_requests, request_assignments
    )
    return float(sum(fm + lm for fm, lm in agg.values()))


def _count_used_lines(iter_dir: Path) -> Optional[int]:
    """Number of MT lines / routes with positive frequency in the ILP solution (``used_lines.csv``)."""
    path = iter_dir / "used_lines.csv"
    if not path.is_file():
        return None
    return len(pd.read_csv(path, encoding="utf-8-sig"))


def _count_passengers_mod_vs_mt_served(iter_dir: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    From ``passenger_assignments.csv``: MoD-only (``no_MT``), MT (assigned route / line),
    and all served (not rejected / not dropped).

    Rows with ``rejected`` or ``Dropped`` are excluded from all three (served = mod_only + mt).
    """
    path = iter_dir / "passenger_assignments.csv"
    if not path.is_file():
        return None, None, None
    line_val = pd.read_csv(path, encoding="utf-8-sig", usecols=["line"])["line"].astype(str).str.strip()
    mod_only = int((line_val == "no_MT").sum())
    excluded = line_val.isin(("rejected", "Dropped", ""))
    mt = int((~excluded & (line_val != "no_MT")).sum())
    served = mod_only + mt
    return mod_only, mt, served


def _load_mod_only_baseline_darp_cost(mod_only_dir: Path) -> Optional[float]:
    """
    Total DARP objective from the MoD-only baseline: read root ``cost`` from
    ``config.yaml-solution.json`` only (no ``passenger_assignments.csv`` / line-planning leg logic).
    """
    mod_only_dir = Path(mod_only_dir).resolve()
    if not mod_only_dir.is_dir():
        raise FileNotFoundError(f"MoD-only DARP results directory not found: {mod_only_dir}")
    solution_path = mod_only_dir / "config.yaml-solution.json"
    if not solution_path.is_file():
        logging.warning("MoD-only baseline solution missing: %s", solution_path)
        return None
    try:
        data = json.loads(solution_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logging.error("Could not read MoD-only solution JSON %s: %s", solution_path, e)
        return None
    if data.get("feasible") is False:
        logging.warning("MoD-only baseline solution is infeasible: %s", solution_path)
        return None
    cost = data.get("cost")
    if cost is None:
        logging.warning("MoD-only solution JSON has no top-level 'cost' key: %s", solution_path)
        return None
    value = float(cost)
    logging.info("MoD-only DARP cost loaded: %s", value)
    return value


def _iter_dirs_sorted(results_dir: Path) -> List[Tuple[int, Path]]:
    pat = re.compile(r"^iteration_(\d+)$")
    out: List[Tuple[int, Path]] = []
    for p in results_dir.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def collect_iteration_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for iteration_index, iter_dir in _iter_dirs_sorted(results_dir):
        metrics_path = iter_dir / "metrics.json"
        mod_est: Optional[float] = None
        line_cost: Optional[float] = None
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            mod_est = payload.get("mod_cost")
            line_cost = payload.get("line_cost")
            if mod_est is not None:
                mod_est = float(mod_est)
            if line_cost is not None:
                line_cost = float(line_cost)

        mod_real: Optional[float] = None
        try:
            mod_real = _sum_mod_real_from_darp_iteration(iter_dir)
        except Exception as e:
            _PLOG.exception("iteration %s: real MoD cost computation failed", iteration_index)
            n_lines = _count_used_lines(iter_dir)
            n_mod_p, n_mt_p, n_served_p = _count_passengers_mod_vs_mt_served(iter_dir)
            rows.append({
                "iteration": iteration_index,
                "mod_cost_estimated": mod_est,
                "line_cost": line_cost,
                "mod_cost_real": None,
                "total_cost_real_plus_line": None,
                "num_lines_used": n_lines,
                "n_passengers_mod_only": n_mod_p,
                "n_passengers_mt": n_mt_p,
                "n_passengers_served": n_served_p,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        total: Optional[float] = None
        if mod_real is not None:
            mod_real_f = float(mod_real)
            if not math.isfinite(mod_real_f):
                _PLOG.warning(
                    "iteration %s: mod_cost_real is not finite (%r); omitting from plot",
                    iteration_index,
                    mod_real_f,
                )
                mod_real = None
            else:
                mod_real = mod_real_f
        if mod_real is not None and line_cost is not None:
            total = float(mod_real + line_cost)
            if not math.isfinite(total):
                _PLOG.warning(
                    "iteration %s: total_cost_real_plus_line is not finite (%r); omitting",
                    iteration_index,
                    total,
                )
                total = None

        if _plot_iteration_costs_debug():
            _PLOG.info(
                "iteration %s: mod_est=%r mod_real=%r line_cost=%r total=%r",
                iteration_index,
                mod_est,
                mod_real,
                line_cost,
                total,
            )

        n_lines = _count_used_lines(iter_dir)
        n_mod_p, n_mt_p, n_served_p = _count_passengers_mod_vs_mt_served(iter_dir)
        rows.append({
            "iteration": iteration_index,
            "mod_cost_estimated": mod_est,
            "line_cost": line_cost,
            "mod_cost_real": mod_real,
            "total_cost_real_plus_line": total,
            "num_lines_used": n_lines,
            "n_passengers_mod_only": n_mod_p,
            "n_passengers_mt": n_mt_p,
            "n_passengers_served": n_served_p,
            "error": None,
        })
    return rows


def build_cost_figure(
    results_dir: Path,
    title: str | None = None,
    *,
    mod_only_results_dir: Path | None = None,
) -> go.Figure:
    results_dir = Path(results_dir).resolve()

    # Load MoD-only baseline cost
    if mod_only_results_dir is not None:
        mod_only_dir = Path(mod_only_results_dir).resolve()
    else:
        mod_only_dir = results_dir.parent / "MoD-only"
    mod_only_baseline_cost = _load_mod_only_baseline_darp_cost(mod_only_dir)
    
    rows = collect_iteration_rows(results_dir)
    if not rows:
        raise FileNotFoundError(
            f"No iteration_* subdirectories with data under {results_dir}"
        )

    if _plot_iteration_costs_debug():
        _PLOG.info("Collected rows:\n%s", pd.DataFrame(rows).to_string())

    display_title = title if title is not None else f"Cost evolution — {results_dir.name}"

    x = [r["iteration"] for r in rows]
    y_mod_est = [r["mod_cost_estimated"] for r in rows]
    y_line = [r["line_cost"] for r in rows]
    y_mod_real = [r["mod_cost_real"] for r in rows]
    y_total = [r["total_cost_real_plus_line"] for r in rows]
    y_n_lines = [r.get("num_lines_used") for r in rows]
    y_n_mod = [r.get("n_passengers_mod_only") for r in rows]
    y_n_mt = [r.get("n_passengers_mt") for r in rows]
    y_n_served = [r.get("n_passengers_served") for r in rows]    

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            "Costs",
            "Lines used (ILP)",
            "Passengers: MoD-only vs MT vs served",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mod_est,
            mode="lines+markers",
            name="MoD cost (estimated, ILP)",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_line,
            mode="lines+markers",
            name="Line cost (ILP)",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mod_real,
            mode="lines+markers",
            name="MoD cost (real, DARP)",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_total,
            mode="lines+markers",
            name="Total = MoD real + line cost",
            line=dict(width=3),
            connectgaps=False,
        ),
        row=1,
        col=1,
    )

    if mod_only_baseline_cost is not None and x:
        x_span = [min(x), max(x)]
        fig.add_trace(
            go.Scatter(
                x=x_span,
                y=[mod_only_baseline_cost, mod_only_baseline_cost],
                mode="lines",
                name="MoD-only baseline (DARP total)",
                line=dict(dash="dash", width=2),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_n_lines,
            mode="lines+markers",
            name="Number of lines used",
            connectgaps=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_n_mod,
            mode="lines+markers",
            name="Passengers MoD-only (no_MT)",
            connectgaps=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_n_mt,
            mode="lines+markers",
            name="Passengers using MT",
            connectgaps=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_n_served,
            mode="lines+markers",
            name="Passengers served (not rejected)",
            connectgaps=False,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=display_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Cost", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Passengers", row=3, col=1)
    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    return fig


# %% Run — build figure, optionally save HTML and show
_ensure_plot_iteration_costs_debug_logging()
fig = build_cost_figure(
    RESULTS_DIR,
    title=FIGURE_TITLE,
    mod_only_results_dir=MOD_ONLY_RESULTS_DIR,
)

if OUTPUT_HTML is not None:
    out_path = Path(OUTPUT_HTML).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    print(f"Wrote {out_path}")

if SHOW_IN_BROWSER:
    # Plain scripts (VS Code "Run Python File", terminal) may otherwise pick a
    # notebook/VS Code renderer and dump HTML to stdout instead of opening a browser.
    pio.renderers.default = "browser"
    fig.show()

# %%
