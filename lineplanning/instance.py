import json
import logging
import random
import re
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple

import numpy as np
import pandas as pd
from pandas.io.sas.sas_constants import dataset_length

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
        dm_file=None
    ):
        self.B = None
        self.candidate_set_of_lines = None  # candidate_set_of_lines[l] contains the nodes served by line l (only useful when building instance from real network)
        self.lengths_travel_times = None  # used only for the manhattan instance
        self.capacity = capacity
        self.demand_file = demand_file
        self.optimal_trip_options: List[List[TripOption]] = []
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

    def _aggregates_from_line_trip_options(
        self,
        optimal_trip_options_per_line: List[List[TripOption]],
        candidate_set_of_lines,
        nb_pass: int,
        nb_lines: int,
    ) -> Tuple[list, list, list, List[List[List[int]]]]:
        """
        Build set_of_lines, pass_to_lines, lines_to_passengers, edge_to_passengers from
        per-line trip options only (no synthetic no-MT row), matching preprocessing().
        """
        pass_to_lines = [list(range(nb_lines)) for _ in range(nb_pass)]
        lines_to_passengers = [list(range(nb_pass)) for _ in range(nb_lines)]
        set_of_lines = []
        edge_to_passengers: List[List[List[int]]] = []
        for line in range(nb_lines):
            line_length = len(candidate_set_of_lines[line]) - 1
            pass_covered = []
            edge_to_passengers.append([[] for _ in range(line_length)])
            for p in range(nb_pass):
                optimal_trip_option = optimal_trip_options_per_line[p][line]
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

    def _load_preprocessing_cache_legacy_json(self, json_path: Path):
        try:
            with json_path.open("r", encoding="utf-8") as cache_file:
                data = json.load(cache_file)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Failed to load preprocessing cache %s: %s", json_path, exc)
            return None

        try:
            set_of_lines = data["set_of_lines"]
            pass_to_lines = data["pass_to_lines"]
            optimal_trip_options_raw = data["optimal_trip_options"]
            optimal_trip_options = [
                [self._deserialize_trip_option(opt) for opt in options]
                for options in optimal_trip_options_raw
            ]
            lines_to_passengers = data["lines_to_passengers"]
            edge_to_passengers = data["edge_to_passengers"]
        except KeyError as exc:
            logging.warning("Missing key in preprocessing cache %s: %s", json_path, exc)
            return None

        logging.info("Loaded preprocessing data from legacy JSON cache %s", json_path)
        return (
            set_of_lines,
            pass_to_lines,
            optimal_trip_options,
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
            optimal_trip_options: List[List[TripOption]] = [[] for _ in range(nb_pass)]
            try:
                col_zip = zip(
                    df["passenger_idx"].to_numpy(copy=False),
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
                    value,
                    mt_pickup_node,
                    mt_drop_off_node,
                    mt_pickup_line_edge_index,
                    mt_drop_off_line_edge_index,
                    first_mile_cost,
                    last_mile_cost,
                    mt_cost,
                ) in tqdm(col_zip, total=len(df), desc="Loading preprocessing cache"):
                    optimal_trip_options[int(p)].append(
                        TripOption(
                            float(value),
                            int(mt_pickup_node),
                            int(mt_drop_off_node),
                            int(mt_pickup_line_edge_index),
                            int(mt_drop_off_line_edge_index),
                            float(first_mile_cost),
                            float(last_mile_cost),
                            float(mt_cost),
                        )
                    )
            except (ValueError, TypeError, KeyError) as exc:
                logging.warning("Invalid row in preprocessing cache %s: %s", cache_path, exc)
                return None

            for p in range(nb_pass):
                if len(optimal_trip_options[p]) != nb_lines:
                    logging.warning(
                        "Preprocessing cache %s: expected %d rows per passenger, got %d for p=%d",
                        cache_path,
                        nb_lines,
                        len(optimal_trip_options[p]),
                        p,
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
            for p in range(nb_pass):
                travel_time = distances[passengers[p][0]][passengers[p][1]]
                optimal_trip_options[p].append(
                    TripOption(0, -1, -1, -1, -1, travel_time, 0, 0)
                )

            logging.info("Loaded preprocessing data from cache %s", cache_path)
            return (
                set_of_lines,
                pass_to_lines,
                optimal_trip_options,
                lines_to_passengers,
                edge_to_passengers,
            )

        legacy_json = cache_path.with_suffix(".json")
        if legacy_json.exists():
            return self._load_preprocessing_cache_legacy_json(legacy_json)

        return None

    def _save_preprocessing_cache(
        self,
        cache_path: Path,
        optimal_trip_options: List[List[TripOption]],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for p, options in enumerate(optimal_trip_options):
            for line_idx, opt in enumerate(options[:-1]):
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
        list, list, List[List[TripOption]], list, List[List[List[int]]], list, list, np.ndarray, List[List[int]]]:
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
                lines_to_passengers,
                edge_to_passengers,
            ) = cached_preprocessing
        else:
            (
                set_of_lines,
                pass_to_lines,
                optimal_trip_options,
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

        return (
            set_of_lines,
            pass_to_lines,
            optimal_trip_options,
            lines_to_passengers,
            edge_to_passengers,
            candidate_set_of_lines,
            lengths_travel_times,
            distances,
            passengers
        )

    def preprocessing(
        self, candidate_set_of_lines, passengers: List[List[int]], travel_times_on_lines, distances, maximum_detour, nb_pass
    ) -> Tuple[list, list, List[List[TripOption]], list, List[List[List[int]]]]:
        logging.info('Preprocessing optimal trip options')
        nb_lines = len(candidate_set_of_lines)

        optimal_trip_options: List[List[TripOption]] = [[] for _ in range(nb_pass)]

        for line in tqdm(range(nb_lines), desc='Processing lines'):
            for p in range(nb_pass):
                optimal_trip_option = self.get_optimal_trip(
                    passengers[p], candidate_set_of_lines[line], travel_times_on_lines[line], distances, maximum_detour
                )
                optimal_trip_options[p].append(optimal_trip_option)

        (
            set_of_lines,
            pass_to_lines,
            lines_to_passengers,
            edge_to_passengers,
        ) = self._aggregates_from_line_trip_options(
            optimal_trip_options, candidate_set_of_lines, nb_pass, nb_lines
        )

        # add the no MT option for each request. The MoD cost is stored in the first_mile_cost field of TripOption.
        for p in range(nb_pass):
            travel_time = distances[passengers[p][0]][passengers[p][1]]
            optimal_trip_options[p].append(TripOption(0, -1, -1, -1, -1, travel_time, 0, 0))

        logging.info('Preprocessing finished')

        return set_of_lines, pass_to_lines, optimal_trip_options, lines_to_passengers, edge_to_passengers

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
            for drop_off_index in range(pickup_index + 1, line_length):
                first_mile_travel_time = distances[origin][line[pickup_index]]
                last_mile_travel_time = distances[line[drop_off_index]][destination]
                mod_travel_time = first_mile_travel_time + last_mile_travel_time
                mt_travel_time = travel_times_on_line[pickup_index][drop_off_index]
                total_travel_time = mod_travel_time + mt_travel_time
                value = int(shortest_travel_time) - int(mod_travel_time)
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




