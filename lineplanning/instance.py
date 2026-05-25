import json
import logging
import random
import re
import hashlib
from array import array
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm

from lineplanning.graph_class import *
from darpinstances.travel_time_provider import MatrixTravelTimeProvider

# , geopandas as gpd

import networkx as nx

import lineplanning.log


# ox.config(log_console=True, use_cache=True)


_REQUESTS_CSV_REQUIRED_COLUMNS = ("origin", "destination", "time")
_REQUESTS_CSV_OPTIONAL_COLUMNS = ("id",)


def _strict_numeric_column(
    df: pd.DataFrame,
    column: str,
    dtype,
    demand_file: Path,
    integral: bool = False,
) -> pd.Series:
    try:
        values = pd.to_numeric(df[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} in {demand_file} must be numeric: {exc}")

    numeric = values.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Column {column!r} in {demand_file} contains NaN or infinite values")
    if integral and not np.equal(numeric, np.trunc(numeric)).all():
        raise ValueError(f"Column {column!r} in {demand_file} must contain integer values")
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        if ((numeric < info.min) | (numeric > info.max)).any():
            raise ValueError(
                f"Column {column!r} in {demand_file} must be in [{info.min}, {info.max}] for {dtype}"
            )
    try:
        return values.astype(dtype, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} in {demand_file} cannot be converted to {dtype}: {exc}")


def _load_demand_from_csv(demand_file: Path) -> pd.DataFrame:
    """
    Load demand data from a Ridesharing_DARP_instances requests.csv file.
    
    Args:
        demand_file: Path to the CSV file
        
    Returns:
        DataFrame with strict ``origin``, ``destination`` and ``time`` columns.
    """
    logging.info('Loading demand from CSV file %s', demand_file)
    try:
        df = pd.read_csv(demand_file)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file {demand_file}: {exc}")

    if df.columns.has_duplicates:
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Invalid requests CSV columns in {demand_file}: duplicate columns {duplicated}")

    allowed_columns = set(_REQUESTS_CSV_REQUIRED_COLUMNS) | set(_REQUESTS_CSV_OPTIONAL_COLUMNS)
    actual_columns = list(df.columns)
    missing_columns = [column for column in _REQUESTS_CSV_REQUIRED_COLUMNS if column not in df.columns]
    unexpected_columns = [column for column in actual_columns if column not in allowed_columns]
    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Invalid requests CSV columns in {demand_file}. "
            f"Expected required columns {list(_REQUESTS_CSV_REQUIRED_COLUMNS)} "
            f"and optional columns {list(_REQUESTS_CSV_OPTIONAL_COLUMNS)}; "
            f"missing={missing_columns}, unexpected={unexpected_columns}, actual={actual_columns}"
        )

    demand = pd.DataFrame(
        {
            "origin": _strict_numeric_column(df, "origin", np.int32, demand_file, integral=True),
            "destination": _strict_numeric_column(df, "destination", np.int32, demand_file, integral=True),
            "time": _strict_numeric_column(df, "time", np.uint32, demand_file, integral=True),
        },
        index=df.index,
    )
    if "id" in df.columns:
        demand["id"] = _strict_numeric_column(df, "id", np.int32, demand_file, integral=True)

    logging.info('Loaded %d demand records from CSV', len(demand))
    return demand


# Above this size, demand-file cache keys use byte length only (content MD5 otherwise).
_DEMAND_CONTENT_HASH_MAX_BYTES = 1 << 30


def _demand_file_cache_token(demand_file_path: Path) -> str:
    stat = demand_file_path.stat()
    if stat.st_size > _DEMAND_CONTENT_HASH_MAX_BYTES:
        return f"size:{stat.st_size}"
    digest = hashlib.md5()
    with open(demand_file_path, "rb") as demand_file:
        while True:
            chunk = demand_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"md5:{digest.hexdigest()}"


def _demand_file_cache_label(demand_token: str) -> str:
    if demand_token.startswith("md5:"):
        return f"dem_{demand_token[4:12]}"
    return demand_token.replace(":", "_")


def preprocessing_csv_path(
    preprocessing_dir: Path,
    demand_file: Path,
    candidate_lines_file: Path,
    maximum_detour: Optional[int],
) -> Path:
    """
    Path to the preprocessing CSV cache for the given demand file, candidate lines file,
    maximum detour, and ``preprocessing_dir`` (directory that will hold ``*.csv`` caches,
    typically ``<instance_folder>/preprocessing``).

    The cache filename hash incorporates demand file content (MD5 of bytes, or file size
    when larger than 1 GiB), resolved candidate-lines path, and maximum detour.
    """
    demand_file_path = Path(demand_file).resolve()
    candidate_line_file_path = Path(candidate_lines_file).resolve()

    demand_token = _demand_file_cache_token(demand_file_path)
    candidate_line_file_str = str(candidate_line_file_path)
    maximum_detour_str = str(maximum_detour) if maximum_detour is not None else "none"

    cache_key = f"{demand_token}|{candidate_line_file_str}|{maximum_detour_str}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]

    demand_name = _demand_file_cache_label(demand_token)
    candidate_line_name = candidate_line_file_path.stem
    detour_suffix = maximum_detour if maximum_detour is not None else "none"

    cache_dir = Path(preprocessing_dir)
    filename = f"{demand_name}_{candidate_line_name}_detour_{detour_suffix}_{cache_hash}.csv"
    return cache_dir / filename


def trip_option_pruning_csv_path(
    base_preprocessing_csv: Path,
    pruning_specs: List[Dict[str, Any]],
) -> Path:
    base = Path(base_preprocessing_csv).resolve()
    key = json.dumps(pruning_specs, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.md5(f"{base}|trip_option_pruning_v1|{key}".encode()).hexdigest()[:12]
    return base.with_name(f"{base.stem}_pruned_{short_hash}.csv")


def normalize_trip_option_pruning_specs(
    pruning_specs: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if pruning_specs is None:
        return []
    if not isinstance(pruning_specs, list):
        raise ValueError("trip_option_pruning must be a list of pruning method objects")

    normalized = []
    for idx, raw_spec in enumerate(pruning_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"trip_option_pruning[{idx}] must be an object")
        method = raw_spec.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"trip_option_pruning[{idx}].method must be a non-empty string")
        method = method.strip()

        if method == "mt_time_share":
            if "min_share" not in raw_spec:
                raise ValueError(f"trip_option_pruning[{idx}].min_share is required for mt_time_share")
            min_share = raw_spec["min_share"]
            if isinstance(min_share, bool) or not isinstance(min_share, Real):
                raise ValueError(f"trip_option_pruning[{idx}].min_share must be a float in [0, 1]")
            min_share = float(min_share)
            if not (0.0 <= min_share <= 1.0):
                raise ValueError(f"trip_option_pruning[{idx}].min_share must be in [0, 1], got {min_share}")
            normalized.append({"method": method, "min_share": min_share})
            continue

        if method == "line_mod_aggregate":
            if "cost_coefficient" not in raw_spec:
                raise ValueError(
                    f"trip_option_pruning[{idx}].cost_coefficient is required for line_mod_aggregate"
                )
            cost_coefficient = raw_spec["cost_coefficient"]
            if isinstance(cost_coefficient, bool) or not isinstance(cost_coefficient, Real):
                raise ValueError(f"trip_option_pruning[{idx}].cost_coefficient must be numeric")
            cost_coefficient = float(cost_coefficient)

            rejection_cost = raw_spec.get("rejection_cost")
            if rejection_cost is not None:
                if isinstance(rejection_cost, bool) or not isinstance(rejection_cost, Real):
                    raise ValueError(f"trip_option_pruning[{idx}].rejection_cost must be numeric or null")
                rejection_cost = float(rejection_cost)
            normalized.append(
                {
                    "method": method,
                    "cost_coefficient": cost_coefficient,
                    "rejection_cost": rejection_cost,
                }
            )
            continue

        raise ValueError(f"Unknown trip option pruning method {method!r} at index {idx}")

    return normalized


def prune_trip_options_mt_time_share(
    optimal_trip_options: pd.DataFrame,
    min_share: float,
) -> Tuple[pd.DataFrame, int]:
    if optimal_trip_options.empty:
        return optimal_trip_options.copy(), 0

    mt_time = optimal_trip_options["mt_cost"].to_numpy(dtype=np.float64, copy=False)
    total_time = (
        optimal_trip_options["first_mile_cost"].to_numpy(dtype=np.float64, copy=False)
        + mt_time
        + optimal_trip_options["last_mile_cost"].to_numpy(dtype=np.float64, copy=False)
    )
    share = np.divide(
        mt_time,
        total_time,
        out=np.zeros(len(optimal_trip_options), dtype=np.float64),
        where=total_time > 0.0,
    )
    keep = share >= float(min_share)
    if bool(keep.all()):
        return optimal_trip_options.copy(), 0
    pruned = optimal_trip_options.loc[keep].copy()
    return pruned.reset_index(drop=True), int((~keep).sum())


def prune_trip_options_line_mod_aggregate(
    optimal_trip_options: pd.DataFrame,
    direct_trip_options: pd.DataFrame,
    nb_lines: int,
    line_opening_costs: List[float],
    rejection_cost: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    For each line ρ, let P_ρ be passengers with a trip option on ρ. If
    ``sum_{p in P_ρ} (fm + lm) + line_cost[ρ] > sum_{p in P_ρ} baseline[p]``,
    remove all trip options on ρ.

    ``line_opening_costs[ρ]`` should match ILP line cost at frequency 1, e.g.
    ``cost_coefficient * lengths_travel_times[ρ]``.

    If ``rejection_cost`` is set and positive, baseline[p] is
    ``min(direct_cost[p], rejection_cost)``; otherwise baseline[p] is direct_cost[p].
    """
    rej = None
    if rejection_cost is not None:
        try:
            rej_val = float(rejection_cost)
        except (TypeError, ValueError):
            rej_val = 0.0
        if rej_val > 0:
            rej = rej_val

    if optimal_trip_options.empty:
        return optimal_trip_options.copy(), []

    df = optimal_trip_options
    work = df[["passenger_idx", "line_idx", "first_mile_cost", "last_mile_cost"]].copy()
    work["mod_cost"] = work["first_mile_cost"].to_numpy() + work["last_mile_cost"].to_numpy()
    direct_costs = direct_trip_options["first_mile_cost"].to_numpy(dtype=np.float64, copy=False)
    baselines = direct_costs[work["passenger_idx"].to_numpy(dtype=np.int64, copy=False)]
    if rej is not None:
        baselines = np.minimum(baselines, rej)
    work["baseline"] = baselines

    sum_mod = work.groupby("line_idx", sort=False)["mod_cost"].sum()
    sum_direct = work.groupby("line_idx", sort=False)["baseline"].sum()
    line_costs = np.asarray(line_opening_costs, dtype=np.float64)

    removable = sum_mod.index[
        sum_mod.to_numpy(dtype=np.float64, copy=False)
        + line_costs[sum_mod.index.to_numpy(dtype=np.int64, copy=False)]
        > sum_direct.reindex(sum_mod.index).to_numpy(dtype=np.float64, copy=False)
    ]
    removed_routes = [int(rho) for rho in removable]
    if not removed_routes:
        return optimal_trip_options.copy(), []

    pruned = df.loc[~df["line_idx"].isin(removed_routes)].copy()
    return pruned.reset_index(drop=True), removed_routes


class line_instance:

    # This class represents abstract instance of the line planning problem, which do not require to know the geometry of the underlying network.

    # instance_category = 'manhattan' allows to create line_instance base on the manhattan network and OD matrix based on fhv data for feb, march, april 2018
    # instance_category = 'grid_network' allows to create line_instance from a grid_network and random OD matrix
    # instance_category = 'random' allows to create a random instance without underlying network

    def __init__(
        self,
        candidate_lines_file,
        capacity,
        demand_file,
        maximum_detour=None,
        preprocessing_dir=None,
        dm_file=None,
        trip_option_pruning: Optional[List[Dict[str, Any]]] = None,
    ):
        self.B = None
        self.candidate_set_of_lines = None  # candidate_set_of_lines[l] contains the nodes served by line l (only useful when building instance from real network)
        self.lengths_travel_times = None  # used only for the manhattan instance
        self.capacity = capacity
        self.demand_file = Path(demand_file).resolve()
        self.optimal_trip_options: pd.DataFrame = self._empty_trip_options_df()
        self._trip_option_keys = np.asarray([], dtype=np.int64)
        self._line_position_cache = {}
        self.direct_trip_options: pd.DataFrame = self._empty_trip_options_df()
        self.dm: Optional[np.ndarray] = None  # dm.
        self.demand: pd.DataFrame = pd.DataFrame(columns=list(_REQUESTS_CSV_REQUIRED_COLUMNS))
        if preprocessing_dir is not None:
            self.preprocessing_dir = Path(preprocessing_dir)
        else:
            self.preprocessing_dir = self.demand_file.parent / "preprocessing"
        self.dm_file = Path(dm_file) if dm_file is not None else Path("dm.h5")
        self.nb_pass: Optional[int] = None
        self.trip_option_pruning = normalize_trip_option_pruning_specs(trip_option_pruning)

        # Store candidate line file path
        self.candidate_line_file = Path(candidate_lines_file)
        if not self.candidate_line_file.exists():
            raise FileNotFoundError("Lines file %s does not exist" % self.candidate_line_file)
        
        # Load candidate lines file early to count number of lines
        # We'll actually load the file in manhattan_instance, but count here to set nb_lines early
        logging.info('Counting candidate lines in %s', self.candidate_line_file)
        with open(self.candidate_line_file, 'r') as f:
            nb_lines = sum(1 for line in f if line.strip())  # Count non-empty lines only
        logging.info('Found %s candidate lines in file', nb_lines)
        self.nb_lines = nb_lines
        (
            self.optimal_trip_options,
            self.direct_trip_options,
            self.candidate_set_of_lines,
            self.lengths_travel_times,
            self.dm,
            self.requests,
            self.demand,
        ) = self.manhattan_instance(maximum_detour)
        self._rebuild_trip_option_index()

    def _get_instance_size_label(self, date: Optional[str]) -> str:
        match = re.search(r"(\d+)_percent", self.demand_file.name)
        if match:
            return f"{match.group(1)}_percent"
        return "100_percent"

    _PREPROCESSING_CSV_COLUMNS = [
        "passenger_idx",
        "line_idx",
        "value",
        "mt_pickup_node",
        "mt_drop_off_node",
        "mt_pickup_line_edge_index",
        "mt_drop_off_line_edge_index",
        "first_mile_cost",
        "last_mile_cost",
        "mt_cost",
    ]

    _PREPROCESSING_CSV_DTYPES = {
        "passenger_idx": np.int32,
        "line_idx": np.int32,
        "value": np.float32,
        "mt_pickup_node": np.int32,
        "mt_drop_off_node": np.int32,
        "mt_pickup_line_edge_index": np.int16,
        "mt_drop_off_line_edge_index": np.int16,
        "first_mile_cost": np.float32,
        "last_mile_cost": np.float32,
        "mt_cost": np.float32,
    }

    _PREPROCESSING_ARRAY_TYPECODES = {
        "passenger_idx": "i",
        "line_idx": "i",
        "value": "f",
        "mt_pickup_node": "i",
        "mt_drop_off_node": "i",
        "mt_pickup_line_edge_index": "h",
        "mt_drop_off_line_edge_index": "h",
        "first_mile_cost": "f",
        "last_mile_cost": "f",
        "mt_cost": "f",
    }

    def _new_trip_option_column_buffers(self) -> dict:
        buffers = {
            column: array(self._PREPROCESSING_ARRAY_TYPECODES[column])
            for column in self._PREPROCESSING_CSV_COLUMNS
        }
        for column, buffer in buffers.items():
            expected = np.dtype(self._PREPROCESSING_CSV_DTYPES[column]).itemsize
            if buffer.itemsize != expected:
                raise RuntimeError(
                    f"array.array typecode {self._PREPROCESSING_ARRAY_TYPECODES[column]!r} "
                    f"for {column!r} has itemsize {buffer.itemsize}, expected {expected}"
                )
        return buffers

    def _append_trip_option_column_values(self, buffers: dict, column: str, values) -> None:
        arr = np.asarray(values, dtype=self._PREPROCESSING_CSV_DTYPES[column])
        if arr.size == 0:
            return
        buffers[column].frombytes(np.ascontiguousarray(arr).tobytes())

    def _trip_options_df_from_column_buffers(self, buffers: dict) -> pd.DataFrame:
        if not buffers or len(buffers["passenger_idx"]) == 0:
            return self._empty_trip_options_df()
        columns = {
            column: np.frombuffer(buffers[column], dtype=self._PREPROCESSING_CSV_DTYPES[column])
            for column in self._PREPROCESSING_CSV_COLUMNS
        }
        return pd.DataFrame(columns, columns=self._PREPROCESSING_CSV_COLUMNS)

    def _empty_trip_options_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: pd.Series(dtype=dtype)
                for column, dtype in self._PREPROCESSING_CSV_DTYPES.items()
            },
            columns=self._PREPROCESSING_CSV_COLUMNS,
        )

    def _direct_trip_options_df_from_distances(
        self,
        passengers: np.ndarray,
        distances: np.ndarray,
    ) -> pd.DataFrame:
        nb_pass = len(passengers)
        if nb_pass == 0:
            return self._empty_trip_options_df()
        passenger_idx = np.arange(nb_pass, dtype=self._PREPROCESSING_CSV_DTYPES["passenger_idx"])
        origins = passengers[:, 0].astype(np.int64, copy=False)
        destinations = passengers[:, 1].astype(np.int64, copy=False)
        direct_costs = distances[origins, destinations].astype(
            self._PREPROCESSING_CSV_DTYPES["first_mile_cost"],
            copy=False,
        )
        return pd.DataFrame(
            {
                "passenger_idx": passenger_idx,
                "line_idx": np.full(nb_pass, -1, dtype=self._PREPROCESSING_CSV_DTYPES["line_idx"]),
                "value": np.zeros(nb_pass, dtype=self._PREPROCESSING_CSV_DTYPES["value"]),
                "mt_pickup_node": np.full(nb_pass, -1, dtype=self._PREPROCESSING_CSV_DTYPES["mt_pickup_node"]),
                "mt_drop_off_node": np.full(nb_pass, -1, dtype=self._PREPROCESSING_CSV_DTYPES["mt_drop_off_node"]),
                "mt_pickup_line_edge_index": np.full(
                    nb_pass,
                    -1,
                    dtype=self._PREPROCESSING_CSV_DTYPES["mt_pickup_line_edge_index"],
                ),
                "mt_drop_off_line_edge_index": np.full(
                    nb_pass,
                    -1,
                    dtype=self._PREPROCESSING_CSV_DTYPES["mt_drop_off_line_edge_index"],
                ),
                "first_mile_cost": direct_costs,
                "last_mile_cost": np.zeros(nb_pass, dtype=self._PREPROCESSING_CSV_DTYPES["last_mile_cost"]),
                "mt_cost": np.zeros(nb_pass, dtype=self._PREPROCESSING_CSV_DTYPES["mt_cost"]),
            },
            columns=self._PREPROCESSING_CSV_COLUMNS,
        )

    def _trip_options_df_from_column_chunks(self, chunks: List[dict]) -> pd.DataFrame:
        if not chunks:
            return self._empty_trip_options_df()

        columns = {}
        for column in self._PREPROCESSING_CSV_COLUMNS:
            arrays = [chunk[column] for chunk in chunks if len(chunk[column]) > 0]
            if arrays:
                columns[column] = np.concatenate(arrays).astype(
                    self._PREPROCESSING_CSV_DTYPES[column],
                    copy=False,
                )
            else:
                columns[column] = np.asarray([], dtype=self._PREPROCESSING_CSV_DTYPES[column])
        return pd.DataFrame(columns, columns=self._PREPROCESSING_CSV_COLUMNS)

    def _sort_trip_options_for_lookup(self, trip_options: pd.DataFrame) -> pd.DataFrame:
        if trip_options.empty:
            return trip_options.copy()
        return trip_options.sort_values(
            ["passenger_idx", "line_idx"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _rebuild_trip_option_index(self) -> None:
        self._line_position_cache = {}
        if self.optimal_trip_options.empty:
            self._trip_option_keys = np.asarray([], dtype=np.int64)
            return
        passengers = self.optimal_trip_options["passenger_idx"].to_numpy(dtype=np.int64, copy=False)
        lines = self.optimal_trip_options["line_idx"].to_numpy(dtype=np.int64, copy=False)
        self._trip_option_keys = passengers * int(self.nb_lines) + lines

    def trip_option_position(self, passenger_idx: int, line_idx: int) -> Optional[int]:
        """Zero-based row position of the feasible trip option in ``optimal_trip_options``."""
        if self._trip_option_keys.size == 0:
            return None
        key = int(passenger_idx) * int(self.nb_lines) + int(line_idx)
        pos = int(np.searchsorted(self._trip_option_keys, key))
        if pos >= self._trip_option_keys.size or int(self._trip_option_keys[pos]) != key:
            return None
        return pos

    def has_trip_option_on_line(self, passenger_idx: int, line_idx: int) -> bool:
        return self.trip_option_position(passenger_idx, line_idx) is not None

    def trip_option_row(self, passenger_idx: int, line_idx: int) -> Optional[pd.Series]:
        """Feasible mass-transit option row for (passenger, candidate line), or None."""
        pos = self.trip_option_position(passenger_idx, line_idx)
        if pos is None:
            return None
        return self.optimal_trip_options.iloc[pos]

    def direct_trip_row(self, passenger_idx: int) -> pd.Series:
        return self.direct_trip_options.iloc[int(passenger_idx)]

    def direct_trip_costs(self, passenger_idx: int) -> Tuple[float, float, float]:
        df = self.direct_trip_options
        p = int(passenger_idx)
        return (
            float(df.iat[p, df.columns.get_loc("first_mile_cost")]),
            float(df.iat[p, df.columns.get_loc("last_mile_cost")]),
            float(df.iat[p, df.columns.get_loc("mt_cost")]),
        )

    def direct_trip_mod_cost(self, passenger_idx: int) -> float:
        first_mile_cost, last_mile_cost, _ = self.direct_trip_costs(passenger_idx)
        return first_mile_cost + last_mile_cost

    def set_direct_trip_costs(
        self,
        passenger_idx: int,
        first_mile_cost: float,
        last_mile_cost: float,
        mt_cost: Optional[float] = None,
    ) -> None:
        p = int(passenger_idx)
        if p < 0 or p >= len(self.direct_trip_options):
            raise IndexError(f"passenger_idx out of range for direct trip option: {passenger_idx}")
        df = self.direct_trip_options
        df.iat[p, df.columns.get_loc("first_mile_cost")] = float(first_mile_cost)
        df.iat[p, df.columns.get_loc("last_mile_cost")] = float(last_mile_cost)
        if mt_cost is not None:
            df.iat[p, df.columns.get_loc("mt_cost")] = float(mt_cost)

    def trip_value_on_line(self, passenger_idx: int, line_idx: int) -> float:
        pos = self.trip_option_position(passenger_idx, line_idx)
        if pos is None:
            return 0.0
        return float(self.optimal_trip_options.iat[pos, self.optimal_trip_options.columns.get_loc("value")])

    def trip_mod_cost_on_line(self, passenger_idx: int, line_idx: int) -> float:
        pos = self.trip_option_position(passenger_idx, line_idx)
        if pos is None:
            return 0.0
        df = self.optimal_trip_options
        return float(df.iat[pos, df.columns.get_loc("first_mile_cost")]) + float(
            df.iat[pos, df.columns.get_loc("last_mile_cost")]
        )

    def trip_costs_on_line(self, passenger_idx: int, line_idx: int) -> Tuple[float, float, float]:
        pos = self.trip_option_position(passenger_idx, line_idx)
        if pos is None:
            raise KeyError(f"No trip option for passenger_idx={passenger_idx}, line_idx={line_idx}")
        df = self.optimal_trip_options
        return (
            float(df.iat[pos, df.columns.get_loc("first_mile_cost")]),
            float(df.iat[pos, df.columns.get_loc("last_mile_cost")]),
            float(df.iat[pos, df.columns.get_loc("mt_cost")]),
        )

    def trip_pickup_dropoff_on_line(self, passenger_idx: int, line_idx: int) -> Tuple[int, int]:
        pos = self.trip_option_position(passenger_idx, line_idx)
        if pos is None:
            raise KeyError(f"No trip option for passenger_idx={passenger_idx}, line_idx={line_idx}")
        df = self.optimal_trip_options
        return (
            int(df.iat[pos, df.columns.get_loc("mt_pickup_node")]),
            int(df.iat[pos, df.columns.get_loc("mt_drop_off_node")]),
        )

    def trip_line_edge_indices(self, passenger_idx: int, line_idx: int) -> Tuple[int, int]:
        pos = self.trip_option_position(passenger_idx, line_idx)
        if pos is None:
            raise KeyError(f"No trip option for passenger_idx={passenger_idx}, line_idx={line_idx}")
        df = self.optimal_trip_options
        return (
            int(df.iat[pos, df.columns.get_loc("mt_pickup_line_edge_index")]),
            int(df.iat[pos, df.columns.get_loc("mt_drop_off_line_edge_index")]),
        )

    def set_trip_mod_cost_on_line(
        self,
        passenger_idx: int,
        line_idx: int,
        first_mile_cost: float,
        last_mile_cost: float,
    ) -> None:
        self.set_trip_costs_on_line(passenger_idx, line_idx, first_mile_cost, last_mile_cost)

    def set_trip_costs_on_line(
        self,
        passenger_idx: int,
        line_idx: int,
        first_mile_cost: float,
        last_mile_cost: float,
        mt_cost: Optional[float] = None,
    ) -> None:
        pos = self.trip_option_position(passenger_idx, line_idx)
        if pos is None:
            raise KeyError(f"No trip option for passenger_idx={passenger_idx}, line_idx={line_idx}")
        df = self.optimal_trip_options
        df.iat[pos, df.columns.get_loc("first_mile_cost")] = float(first_mile_cost)
        df.iat[pos, df.columns.get_loc("last_mile_cost")] = float(last_mile_cost)
        if mt_cost is not None:
            df.iat[pos, df.columns.get_loc("mt_cost")] = float(mt_cost)

    def trip_options_for_passenger(self, passenger_idx: int) -> pd.DataFrame:
        if self._trip_option_keys.size == 0:
            return self._empty_trip_options_df()
        start_key = int(passenger_idx) * int(self.nb_lines)
        end_key = (int(passenger_idx) + 1) * int(self.nb_lines)
        lo, hi = np.searchsorted(self._trip_option_keys, [start_key, end_key])
        return self.optimal_trip_options.iloc[int(lo):int(hi)]

    def trip_option_lines_for_passenger(self, passenger_idx: int) -> List[int]:
        if self._trip_option_keys.size == 0:
            return []
        start_key = int(passenger_idx) * int(self.nb_lines)
        end_key = (int(passenger_idx) + 1) * int(self.nb_lines)
        lo, hi = np.searchsorted(self._trip_option_keys, [start_key, end_key])
        return [
            int(line_idx)
            for line_idx in self.optimal_trip_options["line_idx"].iloc[int(lo):int(hi)].to_numpy(copy=False)
        ]

    def line_length(self, line_idx: int) -> int:
        return len(self.candidate_set_of_lines[int(line_idx)]) - 1

    def first_passenger_on_line(self, line_idx: int) -> Optional[int]:
        passengers = self.line_passengers(line_idx)
        if passengers.size == 0:
            return None
        return int(passengers[0])

    def _line_option_positions(self, line_idx: int) -> np.ndarray:
        rho = int(line_idx)
        cached = self._line_position_cache.get(rho)
        if cached is not None:
            return cached
        if self.optimal_trip_options.empty:
            positions = np.asarray([], dtype=np.int32)
        else:
            line_values = self.optimal_trip_options["line_idx"].to_numpy(copy=False)
            positions = np.flatnonzero(line_values == rho)
            if positions.size == 0 or int(positions[-1]) <= np.iinfo(np.int32).max:
                positions = positions.astype(np.int32, copy=False)
        self._line_position_cache[rho] = positions
        return positions

    def line_passengers(self, line_idx: int) -> np.ndarray:
        positions = self._line_option_positions(line_idx)
        if positions.size == 0:
            return np.asarray([], dtype=np.int32)
        passengers = self.optimal_trip_options["passenger_idx"].to_numpy(copy=False)[positions]
        return np.asarray(passengers, dtype=np.int32)

    def edge_passengers(self, line_idx: int, edge_idx: int) -> np.ndarray:
        positions = self._line_option_positions(line_idx)
        if positions.size == 0:
            return np.asarray([], dtype=np.int32)
        df = self.optimal_trip_options
        pickup_edges = df["mt_pickup_line_edge_index"].to_numpy(copy=False)[positions]
        drop_off_edges = df["mt_drop_off_line_edge_index"].to_numpy(copy=False)[positions]
        edge = int(edge_idx)
        mask = (pickup_edges <= edge) & (edge < drop_off_edges)
        if not bool(mask.any()):
            return np.asarray([], dtype=np.int32)
        passengers = df["passenger_idx"].to_numpy(copy=False)[positions[mask]]
        return np.asarray(passengers, dtype=np.int32)

    def _get_preprocessing_cache_path(
        self,
        maximum_detour: Optional[int],
    ) -> Path:
        """
        Generate cache path from demand content, candidate line file, and maximum detour.
        """
        return preprocessing_csv_path(
            self.preprocessing_dir,
            self.demand_file,
            self.candidate_line_file,
            maximum_detour,
        )

    def _load_preprocessing_cache(
        self,
        cache_path: Path,
        passengers: np.ndarray,
        distances: np.ndarray,
        nb_pass: int,
        nb_lines: int,
    ):
        if cache_path.exists():
            logging.info("Loading preprocessing cache from %s", cache_path)
            try:
                df = pd.read_csv(cache_path, dtype=self._PREPROCESSING_CSV_DTYPES)
            except (OSError, ValueError) as exc:
                logging.warning("Failed to read preprocessing cache %s: %s", cache_path, exc)
                return None

            direct_trip_options = self._direct_trip_options_df_from_distances(passengers, distances)

            logging.info("Loaded preprocessing data from cache %s", cache_path)
            return (
                df.loc[:, self._PREPROCESSING_CSV_COLUMNS],
                direct_trip_options,
            )

        return None

    def _save_preprocessing_cache(
        self,
        cache_path: Path,
        optimal_trip_options: pd.DataFrame,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Saving preprocessing cache to %s", cache_path)
        try:
            optimal_trip_options.to_csv(cache_path, index=False)
        except OSError as exc:
            logging.warning("Failed to write preprocessing cache %s: %s", cache_path, exc)
            return
        logging.info("Stored preprocessing data to cache %s", cache_path)

    def _apply_trip_option_pruning(
        self,
        optimal_trip_options: pd.DataFrame,
        direct_trip_options: pd.DataFrame,
        lengths_travel_times: List[float],
        nb_lines: int,
    ) -> pd.DataFrame:
        for spec in self.trip_option_pruning:
            method = spec["method"]
            before = len(optimal_trip_options)
            if method == "mt_time_share":
                optimal_trip_options, removed_count = prune_trip_options_mt_time_share(
                    optimal_trip_options,
                    spec["min_share"],
                )
                logging.info(
                    "MT-time-share prune removed %s/%s trip options with min_share=%s",
                    removed_count,
                    before,
                    spec["min_share"],
                )
            elif method == "line_mod_aggregate":
                line_opening_costs = [
                    spec["cost_coefficient"] * float(lengths_travel_times[l])
                    for l in range(nb_lines)
                ]
                optimal_trip_options, removed_routes = prune_trip_options_line_mod_aggregate(
                    optimal_trip_options,
                    direct_trip_options,
                    nb_lines,
                    line_opening_costs,
                    rejection_cost=spec["rejection_cost"],
                )
                logging.info(
                    "Line-mod-aggregate prune removed %s routes and %s/%s trip options",
                    len(removed_routes),
                    before - len(optimal_trip_options),
                    before,
                )
                if removed_routes:
                    logging.info("Line-mod-aggregate pruned routes: %s", removed_routes)
            else:
                raise AssertionError(f"Unexpected pruning method {method!r}")
        return self._sort_trip_options_for_lookup(optimal_trip_options)

    def manhattan_instance(self, maximum_detour) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        list,
        list,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
    ]:
        # TODO handle the case where remaining stops pop in skeleton method

        logging.info('Loading distance matrix from %s', self.dm_file)
        if not self.dm_file.exists():
            raise FileNotFoundError("Distance matrix file %s does not exist" % self.dm_file)

        travel_time_provider = MatrixTravelTimeProvider.read_from_file(self.dm_file)
        distances = np.asarray(travel_time_provider.dm)
        logging.info('Distance matrix loaded')

        logging.info('Loading demand')
        if self.demand_file.suffix == '.txt':
            rows = []
            time_list = []
            with open(self.demand_file, 'r') as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        raise ValueError(
                            f"Demand text file {self.demand_file} line {line_no} must contain "
                            "origin, destination and time"
                        )
                    time_value = float(parts[2].strip())
                    if not np.isfinite(time_value) or time_value != np.trunc(time_value):
                        raise ValueError(
                            f"Demand text file {self.demand_file} line {line_no} time must be an integer"
                        )
                    if time_value < 0 or time_value > np.iinfo(np.uint32).max:
                        raise ValueError(
                            f"Demand text file {self.demand_file} line {line_no} time must fit uint32"
                        )
                    rows.append([int(float(parts[0].strip())), int(float(parts[1].strip()))])
                    time_list.append(int(time_value))
            demand = pd.DataFrame(
                {
                    "origin": np.asarray([row[0] for row in rows], dtype=np.int32),
                    "destination": np.asarray([row[1] for row in rows], dtype=np.int32),
                    "time": np.asarray(time_list, dtype=np.uint32),
                }
            )
        else:
            demand = _load_demand_from_csv(self.demand_file)

        passengers = demand.loc[:, ["origin", "destination"]].to_numpy(dtype=np.int32, copy=False)
        self.nb_pass = len(passengers)
        logging.info('Demand loaded')

        nb_pass = len(passengers)

        logging.info('Loading candidate lines from %s', self.candidate_line_file)
        with open(self.candidate_line_file, 'r') as f:
            fist_line = f.readline().strip()
            delimiter = ',' if ',' in fist_line else ' '
            f.seek(0)
            my_list = [line.split(delimiter) for line in open(self.candidate_line_file)]

        candidate_set_of_lines = [[int(float(i.strip())) for i in my_list[j]] for j in range(len(my_list))]
        nb_lines = len(candidate_set_of_lines)

        # travel_times_on_line[i][j][k] contains the time to travel from node number j to node number k on line i
        logging.info('Computing travel times for each line')
        travel_times_on_lines = self.compute_travel_times_on_lines(candidate_set_of_lines, distances)
        logging.info('Travel times computed')

        cache_path = self._get_preprocessing_cache_path(maximum_detour)
        loaded_from_pruned_cache = False
        prune_path = None
        if self.trip_option_pruning:
            prune_path = trip_option_pruning_csv_path(cache_path, self.trip_option_pruning)
            if prune_path.exists():
                pruned_bundle = self._load_preprocessing_cache(
                    prune_path,
                    passengers,
                    distances,
                    nb_pass,
                    nb_lines,
                )
                if pruned_bundle is not None:
                    (
                        optimal_trip_options,
                        direct_trip_options,
                    ) = pruned_bundle
                    loaded_from_pruned_cache = True
                    logging.info(
                        "Loaded pruned trip options from %s; skipped base preprocessing cache",
                        prune_path,
                    )
                else:
                    logging.warning(
                        "Failed to load trip-option pruning cache %s; falling back to base cache",
                        prune_path,
                    )

        if not loaded_from_pruned_cache:
            cached_preprocessing = self._load_preprocessing_cache(
                cache_path,
                passengers,
                distances,
                nb_pass,
                nb_lines,
            )
            if cached_preprocessing is not None:
                (
                    optimal_trip_options,
                    direct_trip_options,
                ) = cached_preprocessing
            else:
                (
                    optimal_trip_options,
                    direct_trip_options,
                ) = self.preprocessing(
                    candidate_set_of_lines,
                    passengers,
                    travel_times_on_lines,
                    distances,
                    maximum_detour,
                    nb_pass,
                )
                self._save_preprocessing_cache(cache_path, optimal_trip_options)

        lengths_travel_times = [
            travel_times_on_lines[l][0][len(candidate_set_of_lines[l]) - 1]
            for l in range(len(candidate_set_of_lines))
        ]

        if self.trip_option_pruning and not loaded_from_pruned_cache:
            optimal_trip_options = self._apply_trip_option_pruning(
                optimal_trip_options,
                direct_trip_options,
                lengths_travel_times,
                nb_lines,
            )
            logging.info(
                "Saving pruned trip options after %s pruning steps to %s",
                len(self.trip_option_pruning),
                prune_path,
            )
            self._save_preprocessing_cache(prune_path, optimal_trip_options)

        return (
            optimal_trip_options,
            direct_trip_options,
            candidate_set_of_lines,
            lengths_travel_times,
            distances,
            passengers,
            demand,
        )

    def preprocessing(
        self, candidate_set_of_lines, passengers: np.ndarray, travel_times_on_lines, distances, maximum_detour, nb_pass
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info('Preprocessing optimal trip options')
        nb_lines = len(candidate_set_of_lines)

        optimal_trip_option_buffers = self._new_trip_option_column_buffers()
        passengers_np = np.asarray(passengers, dtype=np.int64)

        for line in tqdm(range(nb_lines), desc='Processing lines'):
            self._append_optimal_trip_columns_for_line(
                optimal_trip_option_buffers,
                line,
                candidate_set_of_lines[line],
                travel_times_on_lines[line],
                passengers_np,
                distances,
                maximum_detour,
            )

        optimal_trip_options = self._sort_trip_options_for_lookup(
            self._trip_options_df_from_column_buffers(optimal_trip_option_buffers)
        )

        direct_trip_options = self._direct_trip_options_df_from_distances(passengers, distances)

        logging.info('Preprocessing finished')

        return (
            optimal_trip_options,
            direct_trip_options,
        )

    def _append_optimal_trip_columns_for_line(
        self,
        buffers: dict,
        line_idx: int,
        line,
        travel_times_on_line,
        passengers: np.ndarray,
        distances: np.ndarray,
        maximum_detour,
        chunk_size: int = 4096,
    ) -> None:
        line_nodes = np.asarray(line, dtype=np.int64)
        line_length = int(line_nodes.size)
        if line_length < 2:
            return

        pickup_indices, drop_off_indices = np.triu_indices(line_length, k=1)
        if pickup_indices.size == 0:
            return

        travel_times = np.asarray(travel_times_on_line)
        mt_pair_costs = travel_times[pickup_indices, drop_off_indices]
        nb_pass = int(passengers.shape[0])

        for chunk_start in range(0, nb_pass, chunk_size):
            chunk_end = min(chunk_start + chunk_size, nb_pass)
            chunk = passengers[chunk_start:chunk_end]
            origins = chunk[:, 0].astype(np.int64, copy=False)
            destinations = chunk[:, 1].astype(np.int64, copy=False)
            direct = distances[origins, destinations]
            direct_int = direct.astype(np.int64, copy=False)

            first_mile = distances[origins[:, None], line_nodes[None, :]]
            last_mile = distances[line_nodes[None, :], destinations[:, None]]
            chunk_len = int(chunk.shape[0])

            best_value = np.zeros(chunk_len, dtype=np.int64)
            best_mt_cost = np.zeros(chunk_len, dtype=np.float64)
            best_pickup_idx = np.full(chunk_len, -1, dtype=np.int32)
            best_drop_off_idx = np.full(chunk_len, -1, dtype=np.int32)
            best_first_mile = np.zeros(chunk_len, dtype=np.float64)
            best_last_mile = np.zeros(chunk_len, dtype=np.float64)

            for pair_pos, pickup_idx in enumerate(pickup_indices):
                drop_off_idx = int(drop_off_indices[pair_pos])
                pickup_idx = int(pickup_idx)
                mt_cost = float(mt_pair_costs[pair_pos])

                fm = first_mile[:, pickup_idx]
                lm = last_mile[:, drop_off_idx]
                mod_travel_time = fm + lm
                value = direct_int - mod_travel_time.astype(np.int64, copy=False)
                total_travel_time = mod_travel_time + mt_cost
                feasible = total_travel_time <= float(maximum_detour) * direct
                better = feasible & (
                    (value > best_value)
                    | ((value == best_value) & (mt_cost < best_mt_cost))
                )
                if not bool(better.any()):
                    continue

                best_value[better] = value[better]
                best_mt_cost[better] = mt_cost
                best_pickup_idx[better] = pickup_idx
                best_drop_off_idx[better] = drop_off_idx
                best_first_mile[better] = fm[better]
                best_last_mile[better] = lm[better]

            valid = best_pickup_idx != -1
            if not bool(valid.any()):
                continue

            valid_pos = np.flatnonzero(valid)
            best_pickups = best_pickup_idx[valid_pos]
            best_drop_offs = best_drop_off_idx[valid_pos]
            n_valid = int(valid_pos.size)
            self._append_trip_option_column_values(buffers, "passenger_idx", chunk_start + valid_pos)
            self._append_trip_option_column_values(buffers, "line_idx", np.full(n_valid, line_idx, dtype=np.int32))
            self._append_trip_option_column_values(buffers, "value", best_value[valid_pos])
            self._append_trip_option_column_values(buffers, "mt_pickup_node", line_nodes[best_pickups])
            self._append_trip_option_column_values(buffers, "mt_drop_off_node", line_nodes[best_drop_offs])
            self._append_trip_option_column_values(buffers, "mt_pickup_line_edge_index", best_pickups)
            self._append_trip_option_column_values(buffers, "mt_drop_off_line_edge_index", best_drop_offs)
            self._append_trip_option_column_values(buffers, "first_mile_cost", best_first_mile[valid_pos])
            self._append_trip_option_column_values(buffers, "last_mile_cost", best_last_mile[valid_pos])
            self._append_trip_option_column_values(buffers, "mt_cost", best_mt_cost[valid_pos])

    def _optimal_trip_column_chunks_for_line(
        self,
        line_idx: int,
        line,
        travel_times_on_line,
        passengers: np.ndarray,
        distances: np.ndarray,
        maximum_detour,
        chunk_size: int = 4096,
    ) -> List[dict]:
        buffers = self._new_trip_option_column_buffers()
        self._append_optimal_trip_columns_for_line(
            buffers,
            line_idx,
            line,
            travel_times_on_line,
            passengers,
            distances,
            maximum_detour,
            chunk_size=chunk_size,
        )
        if len(buffers["passenger_idx"]) == 0:
            return []
        return [
            {
                column: np.frombuffer(buffers[column], dtype=self._PREPROCESSING_CSV_DTYPES[column])
                for column in self._PREPROCESSING_CSV_COLUMNS
            }
        ]

    def _optimal_trip_records_for_line(
        self,
        line_idx: int,
        line,
        travel_times_on_line,
        passengers: np.ndarray,
        distances: np.ndarray,
        maximum_detour,
        chunk_size: int = 4096,
    ) -> List[tuple]:
        records = []
        for chunk in self._optimal_trip_column_chunks_for_line(
            line_idx,
            line,
            travel_times_on_line,
            passengers,
            distances,
            maximum_detour,
            chunk_size=chunk_size,
        ):
            for row in zip(*(chunk[column] for column in self._PREPROCESSING_CSV_COLUMNS)):
                records.append(tuple(row))
        return records

    def compute_travel_times_on_lines(self, candidate_set_of_lines, distances):
        travel_times_on_lines = []

        line = []
        travel_for_one_line = []
        for i in range(len(candidate_set_of_lines)):
            line = candidate_set_of_lines[i]
            travel_for_one_line = [[-1 for j in range(len(line))] for i in range(len(line))]
            for j in range(len(line)):
                travel_time = 0
                travel_for_one_line[j][j] = 0
                for k in range(j + 1, len(line)):
                    travel_time += distances[line[k - 1]][line[k]]
                    travel_for_one_line[j][k] = travel_time
            travel_times_on_lines.append(travel_for_one_line)
        return travel_times_on_lines




