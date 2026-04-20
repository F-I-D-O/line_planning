"""Jupyter-style script: use Run Cell on each `# %%` region (VS Code / Cursor).

Loads outputs from ``scripts/test_mod_aware_unseen_instances.py``:

  TEST_DIR/<instance_slug>/{baseline,proposed}/metrics.json

Run the **Load** cell once after a validation run; use later cells to add tables or plots
without re-reading disk.

**Cost fields** (from ``metrics.json``), in the order used in the summary table:

- **Total cost** — ``line_cost + darp_total_plan_cost_sum`` (MT line cost plus realized MoD
  from DARP for that line plan).
- **Total ILP cost** — ``objective_value`` (full ILP objective after the one-shot solve).
- **Mod cost** — ``darp_total_plan_cost_sum`` (realized MoD cost from the DARP solution:
  sum of plan ``cost`` over DARP vehicles for the MoD requests induced by the line plan).
- **Line cost** — ``line_cost`` (mass-transit line term in that objective).
- **Estimated mod cost** — ``mod_cost`` (MoD term in the ILP objective, i.e. coefficients
  before DARP is run for that shot).

Pairwise difference is **proposed − baseline** (MoD-calibrated run minus first-iteration run).
**Sign convention:** negative delta means lower cost for **proposed** (improvement vs baseline);
positive means higher cost.

**Percent columns** are ``100 × delta / |baseline|`` so the **sign always matches** ``delta``
(negative ⇒ improvement when lower cost is better). Undefined when ``|baseline|`` is ~0.

The **training progress** cell compares the **same** training MoD-aware run only:
**baseline** = ``iteration_1``, **proposed** = last ``iteration_<n>`` under
``TRAINING_MOD_AWARE_RESULTS_DIR``. DARP totals are read from ``metrics.json`` when present,
otherwise from ``config.yaml-solution.json`` in that iteration folder.

Requires: pandas.
"""

from __future__ import annotations

# %% Configuration — edit this cell only
from pathlib import Path

# Root written by test_mod_aware_unseen_instances.py --work-dir
TEST_DIR = Path(
    r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning\Validation\test"
)

# Difference = proposed - baseline. Set True to show baseline - proposed instead.
DELTA_AS_BASELINE_MINUS_PROPOSED = False

# MoD-aware results root for the *training* run (must contain at least one iteration_* folder).
# If None, the script tries TEST_DIR / "summary.json" key "reference_results_dir".
TRAINING_MOD_AWARE_RESULTS_DIR: Path | None = None


# %% Imports and helpers
import json
import math
import re
from typing import Any, Dict, List, Tuple

import darpinstances.inout
import pandas as pd

# One ordered definition for summary rows, delta columns, and pct columns (must stay aligned).
# Each tuple: summary_index_name, delta_col, baseline_col, pct_col, description
_SUMMARY_ROW_SPEC: Tuple[Tuple[str, str, str, str, str], ...] = (
    (
        "total_cost",
        "delta_total",
        "baseline_total",
        "delta_total_pct",
        "line_cost + darp_total_plan_cost_sum",
    ),
    (
        "total_ILP_cost",
        "delta_total_ilp",
        "baseline_total_ilp",
        "delta_total_ilp_pct",
        "objective_value (total ILP cost)",
    ),
    (
        "mod_cost_darp",
        "delta_mod_darp",
        "baseline_mod_darp",
        "delta_mod_darp_pct",
        "darp_total_plan_cost_sum (realized MoD / DARP)",
    ),
    (
        "line_cost",
        "delta_line",
        "baseline_line",
        "delta_line_pct",
        "line_cost (MT lines)",
    ),
    (
        "estimated_mod_cost_ilp",
        "delta_mod_estimated",
        "baseline_mod_estimated",
        "delta_mod_estimated_pct",
        "mod_cost (ILP estimated MoD term)",
    ),
)

# (delta_col, baseline_col, pct_col) for add_delta_columns — order matches _SUMMARY_ROW_SPEC
_DELTA_BASELINE_PCT: Tuple[Tuple[str, str, str], ...] = tuple(
    (r[1], r[2], r[3]) for r in _SUMMARY_ROW_SPEC
)

_BASELINE_DENOM_EPS = 1e-15


def _show_df(df: pd.DataFrame) -> None:
    try:
        display(df)  # type: ignore[name-defined]
    except NameError:
        print(df.to_string())


def _f(x: Any) -> float:
    if x is None:
        return float("nan")
    return float(x)


def _load_metrics(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_validation_instances(test_dir: Path) -> List[str]:
    """Return subdirectory names that have both baseline and proposed metrics."""
    test_dir = Path(test_dir).resolve()
    if not test_dir.is_dir():
        raise NotADirectoryError(test_dir)
    names: List[str] = []
    for child in sorted(test_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "baseline" / "metrics.json").is_file() and (
            child / "proposed" / "metrics.json"
        ).is_file():
            names.append(child.name)
    return names


def load_validation_frame(test_dir: Path) -> pd.DataFrame:
    """
    One row per instance with baseline_* and proposed_* columns for cost measures,
    including ``*_total`` = line + DARP and ``*_total_ilp`` = objective_value.
    """
    test_dir = Path(test_dir).resolve()
    rows: List[Dict[str, Any]] = []
    for name in discover_validation_instances(test_dir):
        base_path = test_dir / name / "baseline" / "metrics.json"
        prop_path = test_dir / name / "proposed" / "metrics.json"
        b = _load_metrics(base_path)
        p = _load_metrics(prop_path)
        b_line = _f(b.get("line_cost"))
        b_darp = _f(b.get("darp_total_plan_cost_sum"))
        p_line = _f(p.get("line_cost"))
        p_darp = _f(p.get("darp_total_plan_cost_sum"))
        rows.append(
            {
                "instance": name,
                "baseline_total": b_line + b_darp,
                "baseline_total_ilp": _f(b.get("objective_value")),
                "baseline_line": b_line,
                "baseline_mod_darp": b_darp,
                "baseline_mod_estimated": _f(b.get("mod_cost")),
                "proposed_total": p_line + p_darp,
                "proposed_total_ilp": _f(p.get("objective_value")),
                "proposed_line": p_line,
                "proposed_mod_darp": p_darp,
                "proposed_mod_estimated": _f(p.get("mod_cost")),
            }
        )
    return pd.DataFrame(rows)


def add_delta_columns(
    df: pd.DataFrame,
    baseline_minus_proposed: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    sign = -1.0 if baseline_minus_proposed else 1.0
    out["delta_total"] = sign * (out["proposed_total"] - out["baseline_total"])
    out["delta_total_ilp"] = sign * (
        out["proposed_total_ilp"] - out["baseline_total_ilp"]
    )
    out["delta_line"] = sign * (out["proposed_line"] - out["baseline_line"])
    out["delta_mod_darp"] = sign * (out["proposed_mod_darp"] - out["baseline_mod_darp"])
    out["delta_mod_estimated"] = sign * (
        out["proposed_mod_estimated"] - out["baseline_mod_estimated"]
    )
    for delta_col, base_col, pct_col in _DELTA_BASELINE_PCT:
        den_abs = out[base_col].astype(float).abs()
        out[pct_col] = (100.0 * out[delta_col] / den_abs).where(
            den_abs > _BASELINE_DENOM_EPS
        )
    return out


def resolve_training_mod_aware_root(
    test_dir: Path,
    configured: Path | None,
) -> Path:
    if configured is not None:
        return Path(configured).resolve()
    summary_path = Path(test_dir).resolve() / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Set TRAINING_MOD_AWARE_RESULTS_DIR or add summary.json with "
            f"reference_results_dir under {test_dir}"
        )
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    ref = summary.get("reference_results_dir")
    if not ref:
        raise KeyError(
            f"{summary_path} has no 'reference_results_dir'; set "
            f"TRAINING_MOD_AWARE_RESULTS_DIR in the config cell."
        )
    return Path(ref).resolve()


def list_mod_aware_iteration_dirs(training_root: Path) -> List[Tuple[int, Path]]:
    """Sorted ``(index, path)`` for ``iteration_<n>`` subfolders."""
    training_root = Path(training_root).resolve()
    numbered: List[Tuple[int, Path]] = []
    for p in training_root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"iteration_(\d+)", p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    return sorted(numbered, key=lambda x: x[0])


def load_iteration_cost_row(iteration_dir: Path, role: str) -> Dict[str, float]:
    """
    Load one iteration's costs as ``baseline_*`` or ``proposed_*`` keys (same layout as
    ``load_validation_frame``). Fills DARP from ``config.yaml-solution.json`` if missing in
    ``metrics.json``.
    """
    if role not in ("baseline", "proposed"):
        raise ValueError(f"role must be 'baseline' or 'proposed', got {role!r}")
    iteration_dir = Path(iteration_dir).resolve()
    metrics_path = iteration_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    m = _load_metrics(metrics_path)
    line = _f(m.get("line_cost"))
    darp = _f(m.get("darp_total_plan_cost_sum"))
    row = {
        f"{role}_total": line + darp,
        f"{role}_total_ilp": _f(m.get("objective_value")),
        f"{role}_line": line,
        f"{role}_mod_darp": darp,
        f"{role}_mod_estimated": _f(m.get("mod_cost")),
    }
    if math.isnan(row[f"{role}_mod_darp"]):
        sol_path = iteration_dir / "config.yaml-solution.json"
        if sol_path.is_file():
            try:
                darp_v = _darp_total_plan_cost_from_solution_json(sol_path)
                row[f"{role}_mod_darp"] = darp_v
                ln = row[f"{role}_line"]
                row[f"{role}_total"] = (
                    (ln + darp_v) if not math.isnan(ln) else float("nan")
                )
            except (OSError, KeyError, TypeError, ValueError):
                pass
    return row


def load_training_progress_frame(training_mod_aware_root: Path) -> pd.DataFrame:
    """
    Single row: training instance, baseline = first iteration folder, proposed = last
    iteration folder under the MoD-aware results root.
    """
    root = Path(training_mod_aware_root).resolve()
    numbered = list_mod_aware_iteration_dirs(root)
    if not numbered:
        return pd.DataFrame()
    first_n, first_dir = numbered[0]
    last_n, last_dir = numbered[-1]
    label = f"{root.name} (iter {first_n} → {last_n})"
    row: Dict[str, Any] = {"instance": label}
    row.update(load_iteration_cost_row(first_dir, "baseline"))
    row.update(load_iteration_cost_row(last_dir, "proposed"))
    return pd.DataFrame([row])


def build_summary_from_delta_frame(df_delta: pd.DataFrame) -> pd.DataFrame:
    """Mean delta and mean pct columns (same layout as the validation summary table)."""
    delta_cols = [r[1] for r in _SUMMARY_ROW_SPEC]
    pct_cols = [r[3] for r in _SUMMARY_ROW_SPEC]
    mean_row = (
        df_delta[delta_cols].mean()
        if not df_delta.empty
        else pd.Series({c: float("nan") for c in delta_cols})
    )
    mean_pct_row = (
        df_delta[pct_cols].mean()
        if not df_delta.empty
        else pd.Series({c: float("nan") for c in pct_cols})
    )
    idx = [r[0] for r in _SUMMARY_ROW_SPEC]
    desc = [r[4] for r in _SUMMARY_ROW_SPEC]
    return pd.DataFrame(
        {
            "mean_delta": [mean_row[c] for c in delta_cols],
            "mean_delta_pct_of_baseline": [mean_pct_row[c] for c in pct_cols],
            "description": desc,
        },
        index=idx,
    )


def _delta_table_columns() -> List[str]:
    """Column order for per-instance delta tables (matches summary row order)."""
    out: List[str] = ["instance"]
    for r in _SUMMARY_ROW_SPEC:
        out.append(r[1])
        out.append(r[3])
    return out


def _darp_total_plan_cost_from_solution_json(solution_path: Path) -> float:
    sol = darpinstances.inout.load_json(solution_path)
    return float(sum(float(p["cost"]) for p in sol["plans"]))


# %% Load all test instances (run once; keeps ``df_raw`` for downstream cells)
df_raw = load_validation_frame(TEST_DIR)
print(f"Loaded {len(df_raw)} instance(s) from {TEST_DIR.resolve()}")
if df_raw.empty:
    print("No pairs baseline/metrics.json + proposed/metrics.json found.")
else:
    _show_df(df_raw)


# %% Per-instance differences (proposed - baseline by default)
df_delta = add_delta_columns(df_raw, baseline_minus_proposed=DELTA_AS_BASELINE_MINUS_PROPOSED)
label = "baseline - proposed" if DELTA_AS_BASELINE_MINUS_PROPOSED else "proposed - baseline"
print(f"Deltas ({label})")
if not df_delta.empty:
    _show_df(df_delta[_delta_table_columns()])


# %% Average differences across instances
_summary = build_summary_from_delta_frame(df_delta)
print(
    f"Mean across {len(df_delta)} instance(s) ({label}):\n",
)
if not df_delta.empty:
    _show_df(_summary)
else:
    print(_summary.to_string())


# %% Training instance only: iteration 1 → last iteration (MoD-aware progress on training data)
_training_root = resolve_training_mod_aware_root(TEST_DIR, TRAINING_MOD_AWARE_RESULTS_DIR)
print(
    f"Training progress on {_training_root} "
    f"(baseline = first iteration folder, proposed = last iteration folder)"
)
try:
    df_raw_training = load_training_progress_frame(_training_root)
    if df_raw_training.empty:
        print("No iteration_* folders found under training results dir; skip training summary.")
    else:
        df_delta_training = add_delta_columns(
            df_raw_training,
            baseline_minus_proposed=DELTA_AS_BASELINE_MINUS_PROPOSED,
        )
        _label_tr = (
            "baseline - proposed (training: iter 1 → last)"
            if DELTA_AS_BASELINE_MINUS_PROPOSED
            else "proposed - baseline (training: iter 1 → last)"
        )
        print(f"Deltas ({_label_tr})")
        _show_df(df_delta_training[_delta_table_columns()])
        _summary_training = build_summary_from_delta_frame(df_delta_training)
        print(
            f"Summary ({_label_tr}); single training run, "
            f"means equal the one row:\n"
        )
        _show_df(_summary_training)
except FileNotFoundError as exc:
    print(f"Skip training progress table: {exc}")
except (KeyError, json.JSONDecodeError) as exc:
    print(f"Skip training progress table: {exc}")

# %%
