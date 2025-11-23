import json
import logging
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple

import h5py
import numpy as np

from tqdm import tqdm

from graph_class import *

# , geopandas as gpd

import networkx as nx

import log


# ox.config(log_console=True, use_cache=True)


class TripOption(NamedTuple):
    value: float
    mt_pickup_node: int
    mt_drop_off_node: int
    mt_pickup_line_edge_index: int
    mt_drop_off_line_edge_index: int
    first_mile_cost: float
    last_mile_cost: float
    mt_cost: float


class line_instance:

    # This class represents abstract instance of the line planning problem, which do not require to know the geometry of the underlying network.

    # instance_category = 'manhattan' allows to create line_instance base on the manhattan network and OD matrix based on fhv data for feb, march, april 2018
    # instance_category = 'grid_network' allows to create line_instance from a grid_network and random OD matrix
    # instance_category = 'random' allows to create a random instance without underlying network

    def __init__(
        self,
        nb_lines,
        nb_pass,
        cost,
        max_length,
        min_length,
        proba,
        capacity,
        detour_factor=None,
        method=0,
        granularity=1,
        date=None,
        demand_file=None
    ):
        self.nb_lines = nb_lines * granularity  # number of lines in the candidate set.
        self.granularity = granularity
        self.B = None
        self.cost = cost
        self.nb_pass = nb_pass  # number of passengers
        self.proba = proba  # probability that a passenger is covered by a line (when generating random instances)
        self.max_length = max_length  # max length of a line (when generating random instances)
        self.min_length = min_length  # min length of a line (when generating random instances)
        self.candidate_set_of_lines = None  # candidate_set_of_lines[l] contains the nodes served by line l (only useful when building instance from real network)
        self.method = method  # method used by the ILP solver.
        self.lengths_travel_times = None  # used only for the manhattan instance
        self.capacity = capacity
        self.date = None
        self.demand_file = demand_file
        self.optimal_trip_options: List[List[TripOption]] = []
        self.dm: Optional[np.ndarray] = None  # dm.
        self.edge_to_passengers: Optional[List[List[List[int]]]] = None

        lines_file_path = Path(f"all_lines_nodes_{nb_lines}_c5.txt")
        if not lines_file_path.exists():
            raise FileNotFoundError("Lines file %s does not exist" % lines_file_path)
        if demand_file is not None:
            logging.info('Checking number of requests in the demand file %s', demand_file)
            self.nb_pass = len(np.loadtxt(demand_file))
            logging.info('%s requests in the demand file', self.nb_pass)
        else:
            if date == 'april':
                self.nb_pass = 13851
            elif date == 'march':
                self.nb_pass = 12301
            elif date == 'feb':
                self.nb_pass = 13847
            else:
                raise NameError('Not a valid date')
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
        ) = self.manhattan_instance(nb_lines, detour_factor, date)

    def _get_instance_size_label(self, date: Optional[str]) -> str:
        if self.demand_file:
            demand_file_name = Path(self.demand_file).name
            match = re.search(r"(\d+)_percent", demand_file_name)
            if match:
                return f"{match.group(1)}_percent"
        return "100_percent"

    @staticmethod
    def _serialize_trip_option(option: TripOption) -> dict:
        return {
            "value": option.value,
            "mt_pickup_node": option.mt_pickup_node,
            "mt_drop_off_node": option.mt_drop_off_node,
            "mt_pickup_line_edge_index": option.mt_pickup_line_edge_index,
            "mt_drop_off_line_edge_index": option.mt_drop_off_line_edge_index,
            "first_mile_cost": option.first_mile_cost,
            "last_mile_cost": option.last_mile_cost,
            "mt_cost": option.mt_cost,
        }

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

    def _get_preprocessing_cache_path(
        self,
        nb_lines: int,
        detour_factor: Optional[int],
        date: Optional[str],
    ) -> Path:
        instance_label = self._get_instance_size_label(date)
        date_suffix = date if date else "april"
        detour_suffix = detour_factor if detour_factor is not None else "none"
        cache_dir = Path("Results") / instance_label / "preprocessing"
        filename = f"manhattan_{date_suffix}_lines_{nb_lines}_detour_{detour_suffix}.json"
        return cache_dir / filename

    def _load_preprocessing_cache(self, cache_path: Path):
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as cache_file:
                data = json.load(cache_file)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Failed to load preprocessing cache %s: %s", cache_path, exc)
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
            logging.warning("Missing key in preprocessing cache %s: %s", cache_path, exc)
            return None

        logging.info("Loaded preprocessing data from cache %s", cache_path)
        return (
            set_of_lines,
            pass_to_lines,
            optimal_trip_options,
            lines_to_passengers,
            edge_to_passengers,
        )

    def _save_preprocessing_cache(
        self,
        cache_path: Path,
        set_of_lines,
        pass_to_lines,
        optimal_trip_options,
        lines_to_passengers,
        edge_to_passengers,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        optimal_trip_options_serialized = [
            [self._serialize_trip_option(opt) for opt in options]
            for options in optimal_trip_options
        ]
        cache_payload = {
            "set_of_lines": set_of_lines,
            "pass_to_lines": pass_to_lines,
            "optimal_trip_options": optimal_trip_options_serialized,
            "lines_to_passengers": lines_to_passengers,
            "edge_to_passengers": edge_to_passengers,
        }
        try:
            with cache_path.open("w", encoding="utf-8") as cache_file:
                json.dump(cache_payload, cache_file)
        except OSError as exc:
            logging.warning("Failed to write preprocessing cache %s: %s", cache_path, exc)
            return
        logging.info("Stored preprocessing data to cache %s", cache_path)

    def manhattan_instance(self, nb_lines, detour_factor, date) -> Tuple[
        list, list, List[List[TripOption]], list, List[List[List[int]]], list, list, np.ndarray, List[List[int]]]:
        # TODO handle the case where remaining stops pop in skeleton method

        logging.info('Loading distance matrix')
        with  h5py.File('dm.h5', 'r') as f:
            distances = np.array(f['dm'])
        logging.info('Distance matrix loaded')

        # load OD_matrix
        # passengers = np.loadtxt(f)
        # my_list = [line.split(' ') for line in open('OD_matrix_feb.txt','r')]
        # my_list = [line.split(' ') for line in open('OD_matrix_random2.txt','r')]
        # my_list = [line.split(' ') for line in open('OD_matrix_feb_fhv.txt','r')]
        # my_list = [line.split(' ') for line in open('OD_matrix_march_fhv.txt','r')]
        logging.info('Loading demand')
        if self.demand_file is not None:
            my_list = [line.split(' ') for line in open(self.demand_file, 'r')]  # use the demand file provided
        else:
            if date == 'april':
                my_list = [line.split(' ') for line in open('OD_matrix_april_fhv.txt', 'r')]
            if date == 'march':
                my_list = [line.split(' ') for line in open('OD_matrix_march_fhv.txt', 'r')]
            if date == 'feb':
                my_list = [line.split(' ') for line in open('OD_matrix_feb_fhv.txt', 'r')]

        passengers: List[List[int]] = [[int(float(i.strip())) for i in my_list[j]] for j in range(len(my_list))]
        logging.info('Demand loaded')

        nb_pass = len(passengers)

        # load candidate set of lines
        # my_list = [line.split(' ') for line in open('all_lines_nodes_{}.txt'.format(nb_lines))]
        # my_list = [line.split(' ') for line in open('all_lines_nodes_{}_c3.txt'.format(nb_lines))]
        # my_list = [line.split(' ') for line in open('all_lines_nodes_{}_c4.txt'.format(nb_lines))]
        my_list = [line.split(' ') for line in open('all_lines_nodes_{}_c5.txt'.format(nb_lines))]

        candidate_set_of_lines = [[int(float(i.strip())) for i in my_list[j]] for j in range(len(my_list))]

        # travel_times_on_line[i][j][k] contains the time to travel from node number j to node number k on line i
        logging.info('Computing travel times for each line')
        travel_times_on_lines = self.compute_travel_times_on_lines(candidate_set_of_lines, distances)
        logging.info('Travel times computed')

        cache_path = self._get_preprocessing_cache_path(nb_lines, detour_factor, date)
        cached_preprocessing = self._load_preprocessing_cache(cache_path)

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
                detour_factor,
                nb_lines,
                nb_pass,
            )
            self._save_preprocessing_cache(
                cache_path,
                set_of_lines,
                pass_to_lines,
                optimal_trip_options,
                lines_to_passengers,
                edge_to_passengers,
            )

        granularity = self.granularity

        candidate_set_of_lines_tmp = []
        for l in range(len(candidate_set_of_lines)):
            for t in range(granularity):
                candidate_set_of_lines_tmp.append(candidate_set_of_lines[l])
        candidate_set_of_lines = candidate_set_of_lines_tmp

        lengths_travel_times = [travel_times_on_lines[l // granularity][0][len(candidate_set_of_lines[l]) - 1] for l in
            range(len(candidate_set_of_lines))]

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
        self, candidate_set_of_lines, passengers: List[List[int]], travel_times_on_lines, distances, detour_factor, nb_lines, nb_pass
    ) -> Tuple[list, list, List[List[TripOption]], list, List[List[List[int]]]]:
        logging.info('Preprocessing optimal trip options')
        set_of_lines = []
        pass_to_lines = [[] for _ in range(nb_pass)]
        value = 0

        # optimal trip option for each passenger-line combination
        optimal_trip_options: List[List[TripOption]] = [[] for _ in range(nb_pass)]

        lines_to_passengers = []

        # for each edge on each line, this collection contains a list of passengers traveling on the line
        edge_to_passengers: List[List[List[int]]] = []

        for line in tqdm(range(nb_lines), desc='Processing lines'):
            line_length = len(candidate_set_of_lines[line]) - 1
            pass_covered = []
            lines_to_passengers.append([])
            edge_to_passengers.append([[] for _ in range(line_length)])
            for p in range(nb_pass):
                optimal_trip_option = self.get_optimal_trip(
                    passengers[p], candidate_set_of_lines[line], travel_times_on_lines[line], distances, detour_factor
                )

                pass_covered.append(
                    [
                        p,
                        [optimal_trip_option.mt_pickup_node, optimal_trip_option.mt_drop_off_node],
                        optimal_trip_option.value
                    ]
                )
                pass_to_lines[p].append(line)
                lines_to_passengers[line].append(p)
                for line_edge in range(optimal_trip_option.mt_pickup_line_edge_index, optimal_trip_option.mt_drop_off_line_edge_index):
                    edge_to_passengers[line][line_edge].append(p)

                optimal_trip_options[p].append(optimal_trip_option)
            set_of_lines.append([line_length, pass_covered])

        # add the no MT option for each request. The MoD cost is stored in the first_mile_cost field of TripOption.
        for p in range(nb_pass):
            travel_time = distances[passengers[p][0]][passengers[p][1]]
            optimal_trip_options[p].append(TripOption(0, -1, -1, -1, -1, travel_time, 0, 0))

        logging.info('Preprocessing finished')

        return set_of_lines, pass_to_lines, optimal_trip_options, lines_to_passengers, edge_to_passengers

    def get_optimal_trip(self, passenger: List[int], line, travel_times_on_line, distances, detour_factor) -> TripOption:
        """
        Return the optimal trip option for a given passenger on a given line (omega_{ell,p}).
        passenger = [origin, destination]
        line = [list of nodes in the line]
        travel_times_on_line[i][j] contains the time to travel from node number i to node number j on the line
        distances[i][j] distance matrix
        detour_factor = maximum detour factor allowed
        Return Mass transit pickup and drop off nodes, and other trip information
        """
        line_length = len(line)
        origin = passenger[0]
        destination = passenger[1]
        shortest_travel_time = distances[origin][destination]

        optimal_trip_option: TripOption = TripOption(0, -1, -1, -1, -1, 0, 0, 0)

        # compute the optimal trip option, consider all possible pairs of enter and exit nodes for MT
        for pickup_index in range(line_length - 1):
            for drop_off_index in range(pickup_index + 1, line_length):
                first_mile_travel_time = distances[origin][line[pickup_index]]
                last_mile_travel_time = distances[line[drop_off_index]][destination]
                mod_travel_time = first_mile_travel_time + last_mile_travel_time
                mt_travel_time = travel_times_on_line[pickup_index][drop_off_index]
                total_travel_time = mod_travel_time + mt_travel_time
                value = shortest_travel_time - mod_travel_time
                if ((
                    value > optimal_trip_option.value and total_travel_time <= detour_factor * shortest_travel_time) or (
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




