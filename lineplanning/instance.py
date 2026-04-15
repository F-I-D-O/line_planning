import json
import logging
import random
import re
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Dict, NamedTuple, Optional, List, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm

from lineplanning.graph_class import *
from darpinstances.travel_time_provider import MatrixTravelTimeProvider

# , geopandas as gpd

import networkx as nx

import lineplanning.log


# ox.config(log_console=True, use_cache=True)


def _load_demand_from_csv(demand_file: Path) -> List[List[str]]:
    """
    Load demand data from a CSV file.
    
    Args:
        demand_file: Path to the CSV file
        
    Returns:
        List of lists where each inner list contains [origin, dest] as strings
    """
    logging.info('Loading demand from CSV file %s', demand_file)
    try:
        df = pd.read_csv(demand_file, delimiter='\t')
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file {demand_file}: {exc}")
    
    # Try to find origin and dest columns (case-insensitive, with common variations)
    origin_col = None
    dest_col = None
    
    # Check for common column name variations
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['origin', 'orig', 'o', 'from', 'pickup', 'pick_up', 'pickup_node']:
            origin_col = col
        elif col_lower in ['dest', 'destination', 'd', 'to', 'dropoff', 'drop_off', 'dropoff_node', 'dest_node']:
            dest_col = col
    
    if origin_col is None:
        raise ValueError(f"Could not find 'origin' column in CSV file {demand_file}. Available columns: {list(df.columns)}")
    if dest_col is None:
        raise ValueError(f"Could not find 'dest' column in CSV file {demand_file}. Available columns: {list(df.columns)}")
    
    logging.info('Found origin column: %s, dest column: %s', origin_col, dest_col)
    
    # Extract origin and dest columns, convert to string, and return as list of lists
    my_list = [[str(row[origin_col]), str(row[dest_col])] for _, row in df.iterrows()]
    
    logging.info('Loaded %d demand records from CSV', len(my_list))
    return my_list


class TripOption(NamedTuple):
    value: float
    mt_pickup_node: int
    mt_drop_off_node: int
    mt_pickup_line_edge_index: int
    mt_drop_off_line_edge_index: int
    first_mile_cost: float
    last_mile_cost: float
    mt_cost: float


def preprocessing_csv_path(
    preprocessing_dir: Path,
    demand_file: Optional[Path],
    candidate_lines_file: Path,
    maximum_detour: Optional[int],
) -> Path:
    """
    Path to the preprocessing CSV cache for the given demand file, candidate lines file,
    maximum detour, and ``preprocessing_dir`` (directory that will hold ``*.csv`` caches,
    typically ``<instance_folder>/preprocessing``).
    """
    demand_file_path = Path(demand_file).resolve() if demand_file is not None else None
    candidate_line_file_path = Path(candidate_lines_file).resolve()

    demand_file_str = str(demand_file_path) if demand_file_path is not None else "none"
    candidate_line_file_str = str(candidate_line_file_path)
    maximum_detour_str = str(maximum_detour) if maximum_detour is not None else "none"

    cache_key = f"{demand_file_str}|{candidate_line_file_str}|{maximum_detour_str}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]

    demand_name = demand_file_path.stem if demand_file_path is not None else "none"
    candidate_line_name = candidate_line_file_path.stem
    detour_suffix = maximum_detour if maximum_detour is not None else "none"

    cache_dir = Path(preprocessing_dir)
    filename = f"{demand_name}_{candidate_line_name}_detour_{detour_suffix}_{cache_hash}.csv"
    return cache_dir / filename


def line_mod_aggregate_prune_csv_path(
    base_preprocessing_csv: Path,
    cost_coefficient: float,
    rejection_cost: Optional[float] = None,
) -> Path:
    """
    Path to the trip-options CSV after ``line mod aggregate`` pruning (see
    :func:`prune_trip_options_line_mod_aggregate`). The hash includes the base cache path
    and cost coefficient so changing either selects a distinct cache file.
    """
    base = Path(base_preprocessing_csv).resolve()
    rej_key = "none" if rejection_cost is None else repr(float(rejection_cost))
    key = f"{base}|line_mod_agg_prune_v2|cc={cost_coefficient!r}|rej={rej_key}"
    short_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    return base.with_name(f"{base.stem}_modagg_pruned_{short_hash}.csv")


def prune_trip_options_line_mod_aggregate(
    optimal_trip_options: List[Dict[int, TripOption]],
    direct_trip_options: List[TripOption],
    nb_lines: int,
    line_opening_costs: List[float],
    rejection_cost: Optional[float] = None,
) -> Tuple[List[Dict[int, TripOption]], List[int]]:
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

    removed_routes: List[int] = []
    for rho in range(nb_lines):
        ps = [p for p in range(len(optimal_trip_options)) if rho in optimal_trip_options[p]]
        if not ps:
            continue
        sum_mod = sum(
            float(optimal_trip_options[p][rho].first_mile_cost)
            + float(optimal_trip_options[p][rho].last_mile_cost)
            for p in ps
        )
        if rej is None:
            sum_direct = sum(float(direct_trip_options[p].first_mile_cost) for p in ps)
        else:
            sum_direct = sum(min(float(direct_trip_options[p].first_mile_cost), rej) for p in ps)
        lc = float(line_opening_costs[rho])
        if sum_mod + lc > sum_direct:
            removed_routes.append(rho)

    remove_set = set(removed_routes)
    pruned: List[Dict[int, TripOption]] = [
        {rho: opt for rho, opt in by_line.items() if rho not in remove_set}
        for by_line in optimal_trip_options
    ]
    return pruned, removed_routes


class line_instance:

    # This class represents abstract instance of the line planning problem, which do not require to know the geometry of the underlying network.

    # instance_category = 'manhattan' allows to create line_instance base on the manhattan network and OD matrix based on fhv data for feb, march, april 2018
    # instance_category = 'grid_network' allows to create line_instance from a grid_network and random OD matrix
    # instance_category = 'random' allows to create a random instance without underlying network

    def __init__(
        self,
        candidate_lines_file,
        capacity,
        maximum_detour=None,
        demand_file=None,
        preprocessing_dir=None,
        dm_file=None,
        line_mod_aggregate_prune: bool = False,
        line_mod_aggregate_prune_cost_coefficient: float = 1.0,
        line_mod_aggregate_prune_rejection_cost: Optional[float] = None,
    ):
        self.B = None
        self.candidate_set_of_lines = None  # candidate_set_of_lines[l] contains the nodes served by line l (only useful when building instance from real network)
        self.lengths_travel_times = None  # used only for the manhattan instance
        self.capacity = capacity
        self.demand_file = demand_file
        self.optimal_trip_options: List[Dict[int, TripOption]] = []
        self.direct_trip_options: List[TripOption] = []
        self.dm: Optional[np.ndarray] = None  # dm.
        self.edge_to_passengers: Optional[List[List[List[int]]]] = None
        if preprocessing_dir is not None:
            self.preprocessing_dir = Path(preprocessing_dir)
        elif demand_file is not None:
            self.preprocessing_dir = Path(demand_file).parent / "preprocessing"
        else:
            self.preprocessing_dir = Path(candidate_lines_file).parent / "preprocessing"
        self.dm_file = Path(dm_file) if dm_file is not None else Path("dm.h5")
        self.nb_pass: Optional[int] = None
        self._line_mod_aggregate_prune = bool(line_mod_aggregate_prune)
        self._line_mod_aggregate_prune_cost_coefficient = float(line_mod_aggregate_prune_cost_coefficient)
        self._line_mod_aggregate_prune_rejection_cost = (
            None
            if line_mod_aggregate_prune_rejection_cost is None
            else float(line_mod_aggregate_prune_rejection_cost)
        )

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
            self.set_of_lines,
            self.pass_to_lines,
            self.optimal_trip_options,
            self.direct_trip_options,
            self.lines_to_passengers,
            self.edge_to_passengers,
            self.candidate_set_of_lines,
            self.lengths_travel_times,
            self.dm,
            self.requests
        ) = self.manhattan_instance(maximum_detour)

    def _get_instance_size_label(self, date: Optional[str]) -> str:
        if self.demand_file:
            demand_file_name = Path(self.demand_file).name
            match = re.search(r"(\d+)_percent", demand_file_name)
            if match:
                return f"{match.group(1)}_percent"
        return "100_percent"

    @staticmethod
    def _deserialize_trip_option(data: dict) -> TripOption:
        return TripOption(
            value=data["value"],
            mt_pickup_node=data["mt_pickup_node"],
            mt_drop_off_node=data["mt_drop_off_node"],
            mt_pickup_line_edge_index=data["mt_pickup_line_edge_index"],
            mt_drop_off_line_edge_index=data["mt_drop_off_line_edge_index"],
            first_mile_cost=data["first_mile_cost"],
            last_mile_cost=data["last_mile_cost"],
            mt_cost=data["mt_cost"],
        )

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

    def trip_option_on_line(self, passenger_idx: int, line_idx: int) -> Optional[TripOption]:
        """Feasible mass-transit option for (passenger, candidate line), or None if infeasible."""
        return self.optimal_trip_options[passenger_idx].get(line_idx)

    def trip_value_on_line(self, passenger_idx: int, line_idx: int) -> float:
        opt = self.trip_option_on_line(passenger_idx, line_idx)
        return float(opt.value) if opt is not None else 0.0

    def _aggregates_from_line_trip_options(
        self,
        optimal_trip_options_per_line: List[Dict[int, TripOption]],
        candidate_set_of_lines,
        nb_pass: int,
        nb_lines: int,
    ) -> Tuple[list, list, list, List[List[List[int]]]]:
        """
        Build set_of_lines, pass_to_lines, lines_to_passengers, edge_to_passengers from
        feasible per-line trip options only (no direct / no-MT row).
        """
        pass_to_lines = [sorted(optimal_trip_options_per_line[p].keys()) for p in range(nb_pass)]
        lines_to_passengers: List[List[int]] = [[] for _ in range(nb_lines)]
        for p in range(nb_pass):
            for rho in optimal_trip_options_per_line[p]:
                lines_to_passengers[rho].append(p)
        for rho in range(nb_lines):
            lines_to_passengers[rho].sort()

        set_of_lines = []
        edge_to_passengers: List[List[List[int]]] = []
        for line in tqdm(range(nb_lines), desc="Building line instance data structures"):
            line_length = len(candidate_set_of_lines[line]) - 1
            pass_covered = []
            edge_to_passengers.append([[] for _ in range(line_length)])
            for p in range(nb_pass):
                optimal_trip_option = optimal_trip_options_per_line[p].get(line)
                if optimal_trip_option is None:
                    continue
                pass_covered.append(
                    [
                        p,
                        [
                            optimal_trip_option.mt_pickup_node,
                            optimal_trip_option.mt_drop_off_node,
                        ],
                        optimal_trip_option.value,
                    ]
                )
                for line_edge in range(
                    optimal_trip_option.mt_pickup_line_edge_index,
                    optimal_trip_option.mt_drop_off_line_edge_index,
                ):
                    edge_to_passengers[line][line_edge].append(p)
            set_of_lines.append([line_length, pass_covered])
        return set_of_lines, pass_to_lines, lines_to_passengers, edge_to_passengers

    def _get_preprocessing_cache_path(
        self,
        maximum_detour: Optional[int],
    ) -> Path:
        """
        Generate cache path based on demand file, candidate line file, and maximum detour.
        Uses hash of file paths to ensure uniqueness.
        """
        return preprocessing_csv_path(
            self.preprocessing_dir,
            Path(self.demand_file) if self.demand_file is not None else None,
            self.candidate_line_file,
            maximum_detour,
        )

    def _load_preprocessing_cache_legacy_json(
        self,
        json_path: Path,
        candidate_set_of_lines,
        nb_pass: int,
        nb_lines: int,
    ):
        try:
            with json_path.open("r", encoding="utf-8") as cache_file:
                data = json.load(cache_file)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Failed to load preprocessing cache %s: %s", json_path, exc)
            return None

        try:
            optimal_trip_options_raw = data["optimal_trip_options"]
        except KeyError as exc:
            logging.warning("Missing key in preprocessing cache %s: %s", json_path, exc)
            return None

        optimal_trip_options: List[Dict[int, TripOption]] = []
        direct_trip_options: List[TripOption] = []
        try:
            for options in optimal_trip_options_raw:
                if len(options) < 2:
                    logging.warning("Invalid passenger entry in legacy cache %s", json_path)
                    return None
                row_list = [self._deserialize_trip_option(opt) for opt in options]
                direct_trip_options.append(row_list[-1])
                by_line: Dict[int, TripOption] = {}
                for rho, opt in enumerate(row_list[:-1]):
                    if opt.mt_pickup_node != -1:
                        by_line[rho] = opt
                optimal_trip_options.append(by_line)
        except (KeyError, TypeError, ValueError) as exc:
            logging.warning("Invalid trip option in legacy cache %s: %s", json_path, exc)
            return None

        if len(optimal_trip_options) != nb_pass:
            logging.warning(
                "Legacy cache %s: expected %d passengers, got %d",
                json_path,
                nb_pass,
                len(optimal_trip_options),
            )
            return None

        (
            set_of_lines,
            pass_to_lines,
            lines_to_passengers,
            edge_to_passengers,
        ) = self._aggregates_from_line_trip_options(
            optimal_trip_options, candidate_set_of_lines, nb_pass, nb_lines
        )

        logging.info("Loaded preprocessing data from legacy JSON cache %s", json_path)
        return (
            set_of_lines,
            pass_to_lines,
            optimal_trip_options,
            direct_trip_options,
            lines_to_passengers,
            edge_to_passengers,
        )

    def _load_preprocessing_cache(
        self,
        cache_path: Path,
        candidate_set_of_lines,
        passengers: List[List[int]],
        distances: np.ndarray,
        nb_pass: int,
        nb_lines: int,
    ):
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
            except (OSError, ValueError) as exc:
                logging.warning("Failed to read preprocessing cache %s: %s", cache_path, exc)
                return None

            missing = set(self._PREPROCESSING_CSV_COLUMNS) - set(df.columns)
            if missing:
                logging.warning(
                    "Preprocessing cache %s missing columns %s", cache_path, sorted(missing)
                )
                return None

            df = df.sort_values(["passenger_idx", "line_idx"])
            optimal_trip_options: List[Dict[int, TripOption]] = [{} for _ in range(nb_pass)]
            try:
                col_zip = zip(
                    df["passenger_idx"].to_numpy(copy=False),
                    df["line_idx"].to_numpy(copy=False),
                    df["value"].to_numpy(copy=False),
                    df["mt_pickup_node"].to_numpy(copy=False),
                    df["mt_drop_off_node"].to_numpy(copy=False),
                    df["mt_pickup_line_edge_index"].to_numpy(copy=False),
                    df["mt_drop_off_line_edge_index"].to_numpy(copy=False),
                    df["first_mile_cost"].to_numpy(copy=False),
                    df["last_mile_cost"].to_numpy(copy=False),
                    df["mt_cost"].to_numpy(copy=False),
                )
                for (
                    p,
                    line_idx,
                    value,
                    mt_pickup_node,
                    mt_drop_off_node,
                    mt_pickup_line_edge_index,
                    mt_drop_off_line_edge_index,
                    first_mile_cost,
                    last_mile_cost,
                    mt_cost,
                ) in tqdm(col_zip, total=len(df), desc="Loading preprocessing cache"):
                    p_i = int(p)
                    rho = int(line_idx)
                    if rho < 0 or rho >= nb_lines or p_i < 0 or p_i >= nb_pass:
                        raise ValueError(f"out-of-range passenger_idx={p_i} or line_idx={rho}")
                    if int(mt_pickup_node) == -1:
                        continue
                    opt = TripOption(
                        float(value),
                        int(mt_pickup_node),
                        int(mt_drop_off_node),
                        int(mt_pickup_line_edge_index),
                        int(mt_drop_off_line_edge_index),
                        float(first_mile_cost),
                        float(last_mile_cost),
                        float(mt_cost),
                    )
                    if rho in optimal_trip_options[p_i]:
                        logging.warning(
                            "Duplicate (passenger_idx=%d, line_idx=%d) in %s; keeping last row",
                            p_i,
                            rho,
                            cache_path,
                        )
                    optimal_trip_options[p_i][rho] = opt
            except (ValueError, TypeError, KeyError) as exc:
                logging.warning("Invalid row in preprocessing cache %s: %s", cache_path, exc)
                return None

            (
                set_of_lines,
                pass_to_lines,
                lines_to_passengers,
                edge_to_passengers,
            ) = self._aggregates_from_line_trip_options(
                optimal_trip_options, candidate_set_of_lines, nb_pass, nb_lines
            )
            direct_trip_options = [
                TripOption(
                    0,
                    -1,
                    -1,
                    -1,
                    -1,
                    float(distances[passengers[p][0]][passengers[p][1]]),
                    0.0,
                    0.0,
                )
                for p in range(nb_pass)
            ]

            logging.info("Loaded preprocessing data from cache %s", cache_path)
            return (
                set_of_lines,
                pass_to_lines,
                optimal_trip_options,
                direct_trip_options,
                lines_to_passengers,
                edge_to_passengers,
            )

        legacy_json = cache_path.with_suffix(".json")
        if legacy_json.exists():
            return self._load_preprocessing_cache_legacy_json(
                legacy_json, candidate_set_of_lines, nb_pass, nb_lines
            )

        return None

    def _save_preprocessing_cache(
        self,
        cache_path: Path,
        optimal_trip_options: List[Dict[int, TripOption]],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for p, by_line in enumerate(optimal_trip_options):
            for line_idx in sorted(by_line.keys()):
                opt = by_line[line_idx]
                rows.append(
                    {
                        "passenger_idx": p,
                        "line_idx": line_idx,
                        "value": opt.value,
                        "mt_pickup_node": opt.mt_pickup_node,
                        "mt_drop_off_node": opt.mt_drop_off_node,
                        "mt_pickup_line_edge_index": opt.mt_pickup_line_edge_index,
                        "mt_drop_off_line_edge_index": opt.mt_drop_off_line_edge_index,
                        "first_mile_cost": opt.first_mile_cost,
                        "last_mile_cost": opt.last_mile_cost,
                        "mt_cost": opt.mt_cost,
                    }
                )
        df = pd.DataFrame(rows, columns=self._PREPROCESSING_CSV_COLUMNS)
        try:
            df.to_csv(cache_path, index=False)
        except OSError as exc:
            logging.warning("Failed to write preprocessing cache %s: %s", cache_path, exc)
            return
        legacy_json = cache_path.with_suffix(".json")
        if legacy_json.exists():
            try:
                legacy_json.unlink()
            except OSError as exc:
                logging.warning("Could not remove legacy cache %s: %s", legacy_json, exc)
        logging.info("Stored preprocessing data to cache %s", cache_path)

    def manhattan_instance(self, maximum_detour) -> Tuple[
        list,
        list,
        List[Dict[int, TripOption]],
        List[TripOption],
        list,
        List[List[List[int]]],
        list,
        list,
        np.ndarray,
        List[List[int]],
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
            my_list = [line.split(' ') for line in open(self.demand_file, 'r')]  # use the demand file provided
        else:
            my_list = _load_demand_from_csv(self.demand_file)

        passengers: List[List[int]] = [[int(float(i.strip())) for i in my_list[j]] for j in range(len(my_list))]
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
        cached_preprocessing = self._load_preprocessing_cache(
            cache_path,
            candidate_set_of_lines,
            passengers,
            distances,
            nb_pass,
            nb_lines,
        )

        if cached_preprocessing is not None:
            (
                set_of_lines,
                pass_to_lines,
                optimal_trip_options,
                direct_trip_options,
                lines_to_passengers,
                edge_to_passengers,
            ) = cached_preprocessing
        else:
            (
                set_of_lines,
                pass_to_lines,
                optimal_trip_options,
                direct_trip_options,
                lines_to_passengers,
                edge_to_passengers,
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

        if self._line_mod_aggregate_prune:
            prune_path = line_mod_aggregate_prune_csv_path(
                cache_path,
                self._line_mod_aggregate_prune_cost_coefficient,
                rejection_cost=self._line_mod_aggregate_prune_rejection_cost,
            )
            if prune_path.exists():
                pruned_bundle = self._load_preprocessing_cache(
                    prune_path,
                    candidate_set_of_lines,
                    passengers,
                    distances,
                    nb_pass,
                    nb_lines,
                )
                if pruned_bundle is None:
                    logging.warning(
                        "Failed to load line-mod-aggregate pruned cache %s; using unpruned trip options",
                        prune_path,
                    )
                else:
                    (
                        set_of_lines,
                        pass_to_lines,
                        optimal_trip_options,
                        direct_trip_options,
                        lines_to_passengers,
                        edge_to_passengers,
                    ) = pruned_bundle
                    logging.info(
                        "Loaded line-mod-aggregate pruned trip options from %s",
                        prune_path,
                    )
            else:
                line_opening_costs = [
                    self._line_mod_aggregate_prune_cost_coefficient * float(lengths_travel_times[l])
                    for l in range(nb_lines)
                ]
                optimal_trip_options, removed_routes = prune_trip_options_line_mod_aggregate(
                    optimal_trip_options,
                    direct_trip_options,
                    nb_lines,
                    line_opening_costs,
                    rejection_cost=self._line_mod_aggregate_prune_rejection_cost,
                )
                if removed_routes:
                    logging.info(
                        "Line-mod-aggregate prune discarded routes %s (saving %s)",
                        removed_routes,
                        prune_path,
                    )
                self._save_preprocessing_cache(prune_path, optimal_trip_options)
                (
                    set_of_lines,
                    pass_to_lines,
                    lines_to_passengers,
                    edge_to_passengers,
                ) = self._aggregates_from_line_trip_options(
                    optimal_trip_options, candidate_set_of_lines, nb_pass, nb_lines
                )

        return (
            set_of_lines,
            pass_to_lines,
            optimal_trip_options,
            direct_trip_options,
            lines_to_passengers,
            edge_to_passengers,
            candidate_set_of_lines,
            lengths_travel_times,
            distances,
            passengers
        )

    def preprocessing(
        self, candidate_set_of_lines, passengers: List[List[int]], travel_times_on_lines, distances, maximum_detour, nb_pass
    ) -> Tuple[list, list, List[Dict[int, TripOption]], List[TripOption], list, List[List[List[int]]]]:
        logging.info('Preprocessing optimal trip options')
        nb_lines = len(candidate_set_of_lines)

        optimal_trip_options: List[Dict[int, TripOption]] = [{} for _ in range(nb_pass)]

        for line in tqdm(range(nb_lines), desc='Processing lines'):
            for p in range(nb_pass):
                optimal_trip_option = self.get_optimal_trip(
                    passengers[p], candidate_set_of_lines[line], travel_times_on_lines[line], distances, maximum_detour
                )
                if optimal_trip_option.mt_pickup_node != -1:
                    optimal_trip_options[p][line] = optimal_trip_option

        (
            set_of_lines,
            pass_to_lines,
            lines_to_passengers,
            edge_to_passengers,
        ) = self._aggregates_from_line_trip_options(
            optimal_trip_options, candidate_set_of_lines, nb_pass, nb_lines
        )

        direct_trip_options = [
            TripOption(
                0,
                -1,
                -1,
                -1,
                -1,
                float(distances[passengers[p][0]][passengers[p][1]]),
                0.0,
                0.0,
            )
            for p in range(nb_pass)
        ]

        logging.info('Preprocessing finished')

        return (
            set_of_lines,
            pass_to_lines,
            optimal_trip_options,
            direct_trip_options,
            lines_to_passengers,
            edge_to_passengers,
        )

    def get_optimal_trip(self, passenger: List[int], line, travel_times_on_line, distances, maximum_detour) -> TripOption:
        """
        Return the optimal trip option for a given passenger on a given line (omega_{ell,p}).
        passenger = [origin, destination]
        line = [list of nodes in the line]
        travel_times_on_line[i][j] contains the time to travel from node number i to node number j on the line
        distances[i][j] distance matrix
        maximum_detour = maximum detour relative to shortest path (see experiment ``mass_transport.maximum_detour``).
        Return Mass transit pickup and drop off nodes, and other trip information
        """
        line_length = len(line)
        origin = passenger[0]
        destination = passenger[1]
        shortest_travel_time = int(distances[origin][destination])

        optimal_trip_option: TripOption = TripOption(0, -1, -1, -1, -1, 0, 0, 0)

        # compute the optimal trip option, consider all possible pairs of enter and exit nodes for MT
        for pickup_index in range(line_length - 1):
            first_mile_travel_time = distances[origin][line[pickup_index]]

            # If the first mile travel time is greater than the direct travel time, there is no reason to consider this trip option
            # The same is true for the case where the value is lower than the current optimal value, since we keep only the best trip
            # option for each passenger-line pair.
            if int(shortest_travel_time) - int(first_mile_travel_time) < optimal_trip_option.value:
                continue

            for drop_off_index in range(pickup_index + 1, line_length):
                last_mile_travel_time = distances[line[drop_off_index]][destination]
                mod_travel_time = first_mile_travel_time + last_mile_travel_time
                value = int(shortest_travel_time) - int(mod_travel_time)

                # if first mile + last mile travel time is greater than the direct travel time, there is no reason to consider this trip option
                # The same is true for the case where the value is lower than the current optimal value, since we keep only the best trip
                # option for each passenger-line pair.
                if value < optimal_trip_option.value:
                    continue

                mt_travel_time = travel_times_on_line[pickup_index][drop_off_index]
                total_travel_time = mod_travel_time + mt_travel_time
                if ((
                    value > optimal_trip_option.value and total_travel_time <= maximum_detour * shortest_travel_time) or (
                    value == optimal_trip_option.value and mt_travel_time < optimal_trip_option.mt_cost)):
                    optimal_trip_option = TripOption(
                        value,
                        line[pickup_index],
                        line[drop_off_index],
                        pickup_index,
                        drop_off_index,
                        first_mile_travel_time,
                        last_mile_travel_time,
                        mt_travel_time
                    )

        return optimal_trip_option

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




