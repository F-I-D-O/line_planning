"""Jupyter-style script: use Run Cell on each `# %%` region (VS Code / Cursor).

Data: ``RESULTS_DIR/iteration_<k>/metrics.json`` (ILP line_cost, mod_cost estimated),
``config.yaml-solution.json`` (+ CSVs) for real MoD, ``used_lines.csv`` for line counts,
and ``passenger_assignments.csv`` for MoD-only vs MT vs served passenger counts.

Optional **MoD-only** folder (default ``<instance>/MoD-only`` next to ``RESULTS_DIR``): same DARP
artifacts as an iteration; draws a horizontal **MoD-only DARP total** on the cost subplot.

Requires: plotly, pandas, darpinstances.
"""

from __future__ import annotations

# %% Configuration — edit this cell only
# %% Imports
import csv
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import darpinstances.inout
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

# MoD-aware results folder containing iteration_1, iteration_2, ...
RESULTS_DIR = Path(
    r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning\Results\manhattan-2_h-50_percent\instance_01\mod-aware-budget"
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


def _load_darp_requests_csv(path: Path) -> List[dict]:
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


def _load_request_assignments_csv(path: Path) -> List[Tuple[str, Optional[int]]]:
    import pandas as pd

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
            request_assignments.append(("line", int(line_value)))
    return request_assignments


def _darp_id_to_line_planning_leg(
    darp_requests: List[dict],
    request_assignments: List[Tuple[str, Optional[int]]],
) -> Dict[int, str]:
    """Match ``_darp_id_to_line_planning_leg`` in MoD-aware_line_selection.py."""
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


def _compute_per_darp_request_costs(
    solution: dict,
    darp_requests: List[dict],
    request_assignments: List[Tuple[str, Optional[int]]],
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


def _aggregate_mod_costs_for_original_requests(
    darp_request_leg_costs: Dict[int, Tuple[float, float]],
    darp_requests: List[dict],
    request_assignments: List[Tuple[str, Optional[int]]],
) -> dict:
    original_request_costs: dict = {}
    for original_id in range(len(request_assignments)):
        kind, line_idx = request_assignments[original_id]
        if kind == "rejected":
            continue
        darp_ids_for_original = [
            req["id"] for req in darp_requests if req["original_request_id"] == original_id
        ]
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


def _sum_mod_real_from_darp_iteration(iter_dir: Path) -> Optional[float]:
    solution_path = iter_dir / "config.yaml-solution.json"
    requests_path = iter_dir / "requests.csv"
    assignments_path = iter_dir / "passenger_assignments.csv"
    if not (solution_path.exists() and requests_path.exists() and assignments_path.exists()):
        return None
    solution = darpinstances.inout.load_json(solution_path)
    darp_requests = _load_darp_requests_csv(requests_path)
    request_assignments = _load_request_assignments_csv(assignments_path)
    darp_request_leg_costs = _compute_per_darp_request_costs(
        solution, darp_requests, request_assignments
    )
    agg = _aggregate_mod_costs_for_original_requests(
        darp_request_leg_costs, darp_requests, request_assignments
    )
    return sum(fm + lm for fm, lm in agg.values())


def _count_used_lines(iter_dir: Path) -> Optional[int]:
    """Number of MT lines / routes with positive frequency in the ILP solution (``used_lines.csv``)."""
    path = iter_dir / "used_lines.csv"
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _count_passengers_mod_vs_mt_served(iter_dir: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    From ``passenger_assignments.csv``: MoD-only (``no_MT``), MT (assigned route / line),
    and all served (not rejected / not dropped).

    Rows with ``rejected`` or ``Dropped`` are excluded from all three (served = mod_only + mt).
    """
    path = iter_dir / "passenger_assignments.csv"
    if not path.is_file():
        return None, None, None
    mod_only = 0
    mt = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_val = str(row.get("line", "")).strip()
            if line_val == "no_MT":
                mod_only += 1
            elif line_val in ("rejected", "Dropped", ""):
                continue
            else:
                mt += 1
    served = mod_only + mt
    return mod_only, mt, served


def _load_mod_only_baseline_darp_cost(mod_only_dir: Path) -> Optional[float]:
    """
    Total realized MoD cost from the MoD-only DARP run (same layout as an ``iteration_*`` folder).
    """
    mod_only_dir = Path(mod_only_dir).resolve()
    if not mod_only_dir.is_dir():
        return None
    try:
        return _sum_mod_real_from_darp_iteration(mod_only_dir)
    except (AssertionError, KeyError, ValueError, TypeError, OSError):
        return None


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
        except (KeyError, ValueError, TypeError) as e:
            n_lines = _count_used_lines(iter_dir)
            n_mod_p, n_mt_p, n_served_p = _count_passengers_mod_vs_mt_served(iter_dir)
            rows.append({
                "iteration": iteration_index,
                "mod_cost_estimated": mod_est,
                "line_cost": line_cost,
                "mod_cost_real": None,
                "num_lines_used": n_lines,
                "n_passengers_mod_only": n_mod_p,
                "n_passengers_mt": n_mt_p,
                "n_passengers_served": n_served_p,
                "error": str(e),
            })
            continue

        total: Optional[float] = None
        if mod_real is not None and line_cost is not None:
            total = mod_real + line_cost

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
    rows = collect_iteration_rows(results_dir)
    if not rows:
        raise FileNotFoundError(
            f"No iteration_* subdirectories with data under {results_dir}"
        )

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

    if mod_only_results_dir is not None:
        mod_only_dir = Path(mod_only_results_dir).resolve()
    else:
        mod_only_dir = results_dir.parent / "MoD-only"
    mod_only_baseline_cost = _load_mod_only_baseline_darp_cost(mod_only_dir)

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
