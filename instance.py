import json
import logging
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple

import numpy as np

from tqdm import tqdm

from graph_class import *

# import osmnx as ox, geopandas as gpd

import networkx as nx

import log


# ox.config(log_console=True, use_cache=True)


class TripOption(NamedTuple):
    value: float
    mt_pickup_node: int
    mt_drop_off_node: int
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
        instance_category='random',
        n=None,
        nb_stops=None,
        detour_factor=None,
        min_start_end_distance=None,
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

        # random instance of the problem
        if instance_category == 'random':
            raise RuntimeError(
                'random instance category not supported right now due o problem modification, use manhattan.'
            )  # self.set_of_lines, self.pass_to_lines, self.optimal_trip_options, self.lines_to_passengers, self.edge_to_passengers = self.random_instance()  # set of lines[i][0]: length of the line. set of lines[i][1]: list of indices of passengers covered by the line  # pass_to_lines[p] contains the list of lines covering passenger p  # values[p][l]: value of passenger p assigned to line l (0 if p not covered)  # lines_to_passengers[l] contains the list of passengers covered by line l

        # instance from grid network
        if instance_category == 'grid':
            raise RuntimeError(
                'grid instance category not supported right now due o problem modification, use manhattan.'
            )  # self.set_of_lines, self.pass_to_lines, self.optimal_trip_options, self.lines_to_passengers, self.edge_to_passengers, self.candidate_set_of_lines = self.random_grid_instance(nb_lines, nb_pass, n, nb_stops, min_length, max_length, detour_factor, min_start_end_distance)

        # instance from Manhattan network
        if instance_category == 'manhattan':
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

    # The function below allow to create instances of the line_planning problem from specific underlying networks

    # random abstract instance of the problem (no underlying network). Each line has some random probability of covering each passenger and there
    # is a random value for each pair (line, passenger)
    def random_instance(self):
        set_of_lines = []
        pass_to_lines = []
        value = 0
        values = []
        lines_to_passengers = []
        edge_to_passengers = []
        for i in range(self.nb_pass):
            pass_to_lines.append([])
            values.append([])
        for l in range(self.nb_lines):
            # length = nb of edges in the line
            length = np.random.randint(self.min_length, self.max_length + 1)
            pass_covered = []
            lines_to_passengers.append([])
            edge_to_passengers.append([[] for k in range(length)])
            for p in range(self.nb_pass):
                covered = np.random.binomial(1, self.proba)
                if covered == 1:
                    i = np.random.randint(length - 1)
                    j = np.random.randint(i + 1, length)
                    value = np.random.uniform(0, 1)
                    pass_covered.append([p, [i, j], value])
                    pass_to_lines[p].append(l)
                    lines_to_passengers[l].append(p)
                    for k in range(i, j):
                        edge_to_passengers[l][k].append(p)
                else:
                    value = 0
                values[p].append(value)
            set_of_lines.append([length, pass_covered])

        return set_of_lines, pass_to_lines, values, lines_to_passengers, edge_to_passengers

    def grid_network(self, n):
        # return a grid of size n*n
        g = Graph(n ** 2)
        for i in range(n):
            for j in range(n):
                if j < n - 1:
                    g.Add_Into_Adjlist(i * n + j, Node_Distance(i * n + j + 1, 1))
                if j > 0:
                    g.Add_Into_Adjlist(i * n + j, Node_Distance(i * n + j - 1, 1))
                if i < n - 1:
                    g.Add_Into_Adjlist(i * n + j, Node_Distance((i + 1) * n + j, 1))
                if i > 0:
                    g.Add_Into_Adjlist(i * n + j, Node_Distance((i - 1) * n + j, 1))
        return g

    def generate_stops(self, nb_stops, nb_nodes):
        # Return a set of nb_stops random stops out of the nb_nodes possible stops
        remaining_nodes = [i for i in range(nb_nodes)]
        stops = []
        for j in range(nb_stops):
            stop_index = random.randint(0, nb_nodes - j - 1)
            new_stop = remaining_nodes.pop(stop_index)
            stops.append(new_stop)
        return stops

    def generate_new_line(self, remaining_stops, distances, length, min_start_end_distance):
        # return a new line, whose stops belong to remaining_stops.
        # print('starting')
        n = len(remaining_stops)
        min_distance = len(distances) // min_start_end_distance
        # distances contains for i,j the length of a shortest path between i,j
        # Chose randomly the initial node and the last node of the line
        start_index = random.randint(0, n - 1)
        start = remaining_stops.pop(start_index)

        distance_start_end = 0
        i = 0
        while i <= 100:
            i += 1
            end_index = random.randint(0, n - 2)
            end = remaining_stops[end_index]
            if distances[start][end] >= min_distance:
                break

        end = remaining_stops.pop(end_index)
        line = [start, end]
        # print(line)

        for i in range(length - 2):
            # print('------')
            # insert stop which leads to the minimum detour
            position = 0
            node_index = 0
            detour = 999999999999
            # Try to insert a new node in position j
            for j in range(1, len(line)):
                for k in range(len(remaining_stops)):
                    det = distances[line[j - 1]][remaining_stops[k]] + distances[line[j]][remaining_stops[k]] - \
                          distances[line[j - 1]][line[j]]
                    # print(j,k, det)
                    if det < detour:
                        position = j
                        detour = det
                        node_index = k
            # print(remaining_stops)
            # print(node_index)
            new_stop = remaining_stops.pop(node_index)
            # print(remaining_stops)
            # print('pos', position)
            line.insert(position, new_stop)
        return line, remaining_stops

    def generate_lines_one_set(self, stops, min_length, max_length, distances, max_nb_lines, min_start_end_distance):
        # return one set of lines built according to the following process: generate a line with nodes in remaining_stops,
        # remove the nodes of the line from remaining stops. Do this until remaining_stops is too small or until we generated enough lines.
        # print('generate set')
        remaining_stops = deepcopy(stops)
        set_of_lines = []
        length = random.randint(min_length, max_length)
        # print('length', length)
        k = 0
        while length < len(remaining_stops) and k < max_nb_lines:
            # print('before', remaining_stops)
            new_line, remaining_stops = self.generate_new_line(
                remaining_stops, distances, length, min_start_end_distance
            )
            # print('after', remaining_stops)
            length = random.randint(min_length, max_length)
            k += 1
            set_of_lines.append(new_line)
        return set_of_lines

    def generate_lines(self, nb_lines, stops, min_length, max_length, distances, min_start_end_distance):
        # return a random candidate set of lines. A line is a list of the indices of the stops.
        all_lines = []
        set_of_lines = []
        while len(all_lines) < nb_lines:
            set_of_lines = self.generate_lines_one_set(
                stops, min_length, max_length, distances, nb_lines - len(all_lines), min_start_end_distance
            )
            all_lines += set_of_lines
        return all_lines

    def generate_lines_skeleton(
        self,
        nb_lines,
        stops,
        min_length,
        max_length,
        distances,
        min_start_end_distance,
        detour_skeleton,
        shortest_paths
    ):
        all_lines = []
        while len(all_lines) < nb_lines:
            length = random.randint(min_length, max_length)
            new_line, remaining_stops = self.generate_new_line_skeleton(
                stops, distances, length, min_start_end_distance, detour_skeleton, shortest_paths
            )
            all_lines.append(new_line)
        return all_lines

    def generate_new_line_skeleton(
        self, stops, distances, length, min_start_end_distance, detour_skeleton, shortest_paths
    ):
        # TODO: pick only nodes which are in shortest_paths inter remaining_stops
        remaining_stops = deepcopy(stops)
        n = len(remaining_stops)
        min_distance = len(distances) // min_start_end_distance
        # distances contains for i,j the length of a shortest path between i,j
        # Chose randomly the initial node and the last node of the line
        start_index = random.randint(0, n - 1)
        start = remaining_stops.pop(start_index)

        distance_start_end = 0
        i = 0
        while i <= 100:
            i += 1
            end_index = random.randint(0, n - 2)
            end = remaining_stops[end_index]
            if distances[start][end] >= min_distance:
                break

        end = remaining_stops.pop(end_index)

        i = 0
        while i <= 100:
            i += 1
            inter_index = random.randint(0, n - 3)
            inter = remaining_stops[inter_index]
            if distances[start][inter] + distances[end][inter] <= distances[start][end] * detour_skeleton:
                break
        inter = remaining_stops.pop(inter_index)

        i = 0
        while i <= 100:
            i += 1
            inter_index_2 = random.randint(0, n - 4)
            inter_2 = remaining_stops[inter_index_2]
            if distances[inter_2][inter] + distances[end][inter_2] <= distances[inter][end] * detour_skeleton:
                break
        inter_2 = remaining_stops.pop(inter_index_2)

        # line = [start, inter, inter_2, end]
        line = shortest_paths[start][inter] + shortest_paths[inter][inter_2][1:] + shortest_paths[inter_2][end][1:]
        while len(line) > length:
            index = random.randint(1, len(line) - 2)
            line.pop(index)
        return line, remaining_stops

    def generate_new_line_skeleton_manhattan(
        self, stops, distances, length, min_start_end_distance, detour_skeleton, G
    ):
        # TODO: pick only nodes which are in shortest_paths inter remaining_stops
        remaining_stops = deepcopy(stops)
        n = len(remaining_stops)
        # min_distance = len(distances)//min_start_end_distance
        min_distance = min_start_end_distance

        # distances contains for i,j the length of a shortest path between i,j
        # Chose randomly the initial node and the last node of the line
        start_index = random.randint(0, n - 1)
        start = remaining_stops.pop(start_index)

        distance_start_end = 0
        i = 0
        while i <= 1000:
            i += 1
            end_index = random.randint(0, n - 2)
            end = remaining_stops[end_index]
            if distances[start][end] >= min_distance:
                break

        end = remaining_stops.pop(end_index)
        # print('ori', start, end, distances[start][end])
        # print('iter', i)

        i = 0
        while i <= 1000:
            i += 1
            inter_index = random.randint(0, n - 3)
            inter = remaining_stops[inter_index]
            if distances[start][inter] + distances[end][inter] <= distances[start][end] * detour_skeleton:
                break
        inter = remaining_stops.pop(inter_index)

        i = 0
        while i <= 1000:
            i += 1
            inter_index_2 = random.randint(0, n - 4)
            inter_2 = remaining_stops[inter_index_2]
            if distances[inter_2][inter] + distances[end][inter_2] <= distances[inter][end] * detour_skeleton:
                break
        inter_2 = remaining_stops.pop(inter_index_2)

        orig = list(G)[start]
        dest = list(G)[end]

        route_within_stops_1, route1 = self.shortest_path_nodes(G, start, inter, stops)
        route_within_stops_2, route2 = self.shortest_path_nodes(G, inter, inter_2, stops)
        route_within_stops_3, route3 = self.shortest_path_nodes(G, inter_2, end, stops)

        line = route_within_stops_1 + route_within_stops_2[1:] + route_within_stops_3[1:]
        route = route1 + route2[1:] + route3[1:]

        while len(line) > length:
            index = random.randint(1, len(line) - 2)
            line.pop(index)
        # print('time_length', distances[start][end])
        # print(line)
        return line, route

    def generate_lines_skeleton_manhattan(
        self, nb_lines, stops, min_length, max_length, distances, min_start_end_distance, detour_skeleton, G
    ):
        all_lines = []
        all_routes = []
        iter = 0
        while len(all_lines) < nb_lines and iter < 2 * nb_lines:
            iter += 1
            if iter % 100 == 0:
                print('iteration', iter)
            try:
                length = random.randint(min_length, max_length)
                new_line, route = self.generate_new_line_skeleton_manhattan(
                    stops, distances, length, min_start_end_distance, detour_skeleton, G
                )
                # print('length', len(new_line))
                all_lines.append(new_line)
                all_routes.append(route)
            except:
                print('line_construction_failed')
        return all_lines, all_routes

    def shortest_path_nodes(self, G, orig_index, dest_index, stops):
        orig = list(G)[orig_index]
        dest = list(G)[dest_index]
        route = nx.shortest_path(G, orig, dest, weight='travel_time')
        # print(route)
        nb = 0
        route_within_stops = []
        for i in range(len(route)):
            for j in range(len(stops)):
                if route[i] == list(G)[stops[j]]:
                    nb += 1
                    route_within_stops.append(route[i])
                    break
        # print(nb)
        # print(route_within_stops)
        # fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')

        return route_within_stops, route

    def random_grid_instance(
        self, nb_lines, nb_pass, n, nb_stops, min_length, max_length, detour_factor, min_start_end_distance
    ):
        # TODO handle the case where remaining stops pop in skeleton method

        nb_nodes = n ** 2
        detour_skeleton = 2
        graph = self.grid_network(n)
        stops = self.generate_stops(nb_stops, nb_nodes)
        # distances[i][j] contains the length of a shortest path between i and j
        distances = [[] for i in range(nb_nodes)]
        shortest_paths = [[] for i in range(nb_nodes)]
        for i in range(nb_nodes):
            dist, short = graph.Dijkstras_Shortest_Path(i)
            distances[i] = dist
            shortest_paths[i] = short
        # generate a random candidate set of lines

        # candidate_set_of_lines = self.generate_lines(nb_lines, stops, min_length, max_length, distances, min_start_end_distance)
        # candidate_set_of_lines = self.generate_lines(nb_lines //2, stops, min_length, max_length, distances, min_start_end_distance) + self.generate_lines_skeleton(nb_lines//2, stops, min_length, max_length, distances, min_start_end_distance, detour_skeleton, shortest_paths)
        candidate_set_of_lines = self.generate_lines_skeleton(
            nb_lines, stops, min_length, max_length, distances, min_start_end_distance, detour_skeleton, shortest_paths
        )

        # travel_times_on_line[i][j][k] contains the time to travel from node number j to node number k on line i
        travel_times_on_lines = self.compute_travel_times_on_lines(candidate_set_of_lines, distances)
        # print('candidate_set_of_lines', candidate_set_of_lines)

        # build the instance
        set_of_lines = []
        pass_to_lines = []
        value = 0
        values = []
        lines_to_passengers = []
        edge_to_passengers = []

        passengers = []

        for p in range(self.nb_pass):
            # for each passenger, compute its starting and end points
            start = random.randint(0, nb_nodes - 1)
            end = random.randint(0, nb_nodes - 1)
            while end == start:
                end = random.randint(0, nb_nodes - 1)
            passenger = [start, end]
            passengers.append(passenger)  # print('passenger', p, passenger)
        # print('passengers', passengers)

        for i in range(self.nb_pass):
            pass_to_lines.append([])
            values.append([])

        for l in range(self.nb_lines):
            # length == nb_edges in the line == nb_stops - 1
            length = len(candidate_set_of_lines[l]) - 1
            pass_covered = []
            lines_to_passengers.append([])
            edge_to_passengers.append([[] for k in range(length)])
            for p in range(self.nb_pass):
                value, enter_node, exit_node = self.get_optimal_trip(
                    passengers[p], candidate_set_of_lines[l], travel_times_on_lines[l], distances, detour_factor
                )
                if value > 0:
                    pass_covered.append([p, [enter_node, exit_node], value])
                    pass_to_lines[p].append(l)
                    lines_to_passengers[l].append(p)
                    for k in range(enter_node, exit_node):
                        edge_to_passengers[l][k].append(p)

                values[p].append(value)
            set_of_lines.append([length, pass_covered])

        return set_of_lines, pass_to_lines, values, lines_to_passengers, edge_to_passengers, candidate_set_of_lines

    def manhattan_instance(self, nb_lines, detour_factor, date) -> Tuple[
        list, list, List[List[TripOption]], list, list, list, list, np.ndarray, List[List[int]]]:
        # TODO handle the case where remaining stops pop in skeleton method

        # load distance matrix for G
        # Distance between each pair of nodes
        logging.info('Loading distance matrix')
        f1 = open("manhattan_dist_1.txt", "r")
        f2 = open("manhattan_dist_2.txt", "r")
        f3 = open("manhattan_dist_3.txt", "r")
        distances = np.loadtxt(f1) + np.loadtxt(f2) + np.loadtxt(f3)
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
    ) -> Tuple[list, list, List[List[TripOption]], list, list]:
        logging.info('Preprocessing optimal trip options')
        set_of_lines = []
        pass_to_lines = [[] for _ in range(nb_pass)]
        value = 0

        # optimal trip option for each passenger-line combination
        optimal_trip_options: List[List[TripOption]] = [[] for _ in range(nb_pass)]

        lines_to_passengers = []
        edge_to_passengers = []

        for l in tqdm(range(nb_lines), desc='Processing lines'):
            # length == nb_edges in the line == nb_stops - 1
            length = len(candidate_set_of_lines[l]) - 1
            pass_covered = []
            lines_to_passengers.append([])
            edge_to_passengers.append([[] for k in range(length)])
            for p in range(nb_pass):
                optimal_trip_option = self.get_optimal_trip(
                    passengers[p], candidate_set_of_lines[l], travel_times_on_lines[l], distances, detour_factor
                )

                pass_covered.append(
                    [
                        p,
                        [optimal_trip_option.mt_pickup_node, optimal_trip_option.mt_drop_off_node],
                        optimal_trip_option.value
                    ]
                )
                pass_to_lines[p].append(l)
                lines_to_passengers[l].append(p)
                for k in range(optimal_trip_option.mt_pickup_node, optimal_trip_option.mt_drop_off_node):
                    edge_to_passengers[l][k].append(p)

                optimal_trip_options[p].append(optimal_trip_option)
            set_of_lines.append([length, pass_covered])

        # add the no MT option for each request. The MoD cost is stored in the first_mile_cost field of TripOption.
        for p in range(nb_pass):
            travel_time = distances[passengers[p][0]][passengers[p][1]]
            optimal_trip_options[p].append(TripOption(0, -1, -1, travel_time, 0, 0))

        logging.info('Preprocessing finished')

        return set_of_lines, pass_to_lines, optimal_trip_options, lines_to_passengers, edge_to_passengers

    def preprocessing_real_time_routing(
        self,
        candidate_set_of_lines,
        passengers,
        travel_times_on_lines,
        distances,
        detour_factor,
        nb_lines,
        nb_pass,
        granularity,
        date
    ):

        # passengers = np.loadtxt(f)
        # my_list = [line.split(' ') for line in open('OD_matrix_feb_pickup_time.txt','r')]
        # my_list = [line.split(' ') for line in open('OD_matrix_random2.txt','r')]
        # my_list = [line.split(' ') for line in open('OD_matrix_feb_fhv.txt','r')]
        # my_list = [line.split(' ') for line in open('OD_matrix_march_fhv.txt','r')]
        # my_list = [line.split(' ') for line in open('OD_matrix_april_fhv.txt','r')]
        if date == 'april':
            my_list = [line.split(' ') for line in open('OD_matrix_april_fhv.txt', 'r')]
        if date == 'march':
            my_list = [line.split(' ') for line in open('OD_matrix_march_fhv.txt', 'r')]
        if date == 'feb':
            my_list = [line.split(' ') for line in open('OD_matrix_feb_fhv.txt', 'r')]

        passengers = [[int(float(i.strip())) for i in my_list[j]] for j in range(len(my_list))]

        '''
        #to remove
        passengers = passengers[0:100]
        nb_pass = len(passengers)
        '''

        # print(passengers)
        request_times = [passengers[i][2] * 60 for i in range(len(passengers))]

        set_of_lines = []
        pass_to_lines = []
        value = 0
        values = []
        lines_to_passengers = []
        edge_to_passengers = []

        for i in range(nb_pass):
            pass_to_lines.append([])
            values.append([])

        print('preprocessing')

        for l in range(nb_lines):
            if l % 100 == 0:
                print(l)
            # length == nb_edges in the line == nb_stops - 1
            length = len(candidate_set_of_lines[l]) - 1

            for t in range(granularity):
                # consider the line departing at time t*60//granularity
                nodes_reach_time = [t * 3600 // granularity + travel_times_on_lines[l][0][j] for j in
                    range(len(candidate_set_of_lines[l]))]
                # travel_times_on_lines[l][0][j] contains the time to go from 0 to j on line l

                pass_covered = []
                lines_to_passengers.append([])
                edge_to_passengers.append([[] for k in range(length)])
                for p in range(nb_pass):
                    value, enter_node, exit_node = self.optimal_trip_real_time_routing(
                        passengers[p],
                        candidate_set_of_lines[l],
                        travel_times_on_lines[l],
                        distances,
                        detour_factor,
                        nodes_reach_time,
                        request_times[p],
                        granularity
                    )
                    if value > 0:
                        # the number of the line is granularity * l + t
                        pass_covered.append([p, [enter_node, exit_node], value])
                        pass_to_lines[p].append(granularity * l + t)
                        lines_to_passengers[granularity * l + t].append(p)
                        for k in range(enter_node, exit_node):
                            edge_to_passengers[granularity * l + t][k].append(p)

                    values[p].append(value)
                set_of_lines.append([length, pass_covered])

        print('--------------------------------------')
        '''print('values', values)
        non_zero_values = []
        for p in range(len(values)):
            non_zero = 0
            for i in range(nb_lines):
                if values[p][i] > 40 :
                    non_zero +=1
            non_zero_values.append(non_zero)
        print('non_zeros', non_zero_values)'''

        return set_of_lines, pass_to_lines, values, lines_to_passengers, edge_to_passengers

    def optimal_trip_real_time_routing(
        self,
        passenger,
        line,
        travel_times_on_line,
        distances,
        detour_factor,
        nodes_reach_time,
        request_time,
        granularity
    ):
        n = len(line)
        # print('lenline', n)
        start = passenger[0]
        end = passenger[1]
        t_car = distances[start][end]
        # print('passenger', start, end)
        # print('t_car', t_car)
        t_bus = 0
        t_wait = 0
        value = 0
        t_trip_car = 0
        enter_node = -1
        exit_node = -1
        # compute the value of the trip
        # print('passenger', start, end, request_time)
        # print('line', line)

        for j in range(n - 1):
            # print(j, 'reached at', nodes_reach_time[j])
            # The trip is possible only if the passenger can reach node j before the bus has reached it
            if distances[start][line[j]] + request_time <= nodes_reach_time[j]:
                # print('passenger_reaches', distances[start][line[j]] + request_time)
                for i in range(j + 1, n):
                    t_trip_car = distances[start][line[j]] + distances[line[i]][end]
                    t_trip_bus = travel_times_on_line[j][i]
                    wait_time = nodes_reach_time[j] - distances[start][line[j]] + request_time
                    t_total = t_trip_car + t_trip_bus + wait_time
                    if wait_time <= 3600 // granularity and (
                        (t_car - t_trip_car > value and t_total <= detour_factor * t_car) or (
                        t_car - t_trip_car == value and t_trip_bus + wait_time < t_wait + t_bus)):
                        # if (t_car - t_trip_car > value and t_total <= detour_factor * t_car) or (t_car - t_trip_car == value and t_trip_bus + wait_time < t_wait + t_bus):
                        value = t_car - t_trip_car
                        enter_node = j
                        exit_node = i
                        t_bus = t_trip_bus
                        t_wait = wait_time
        # print('req', request_time, t_bus, t_wait, t_trip_car)
        # print('value', value)
        return value, enter_node, exit_node

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

        optimal_trip_option: TripOption = TripOption(0, -1, -1, 0, 0, 0)

        # compute the optimal trip option, consider all possible pairs of enter and exit nodes for MT
        for j in range(line_length - 1):
            for i in range(j + 1, line_length):
                first_mile_travel_time = distances[origin][line[j]]
                last_mile_travel_time = distances[line[i]][destination]
                mod_travel_time = first_mile_travel_time + last_mile_travel_time
                mt_travel_time = travel_times_on_line[j][i]
                total_travel_time = mod_travel_time + mt_travel_time
                value = shortest_travel_time - mod_travel_time
                if ((
                    value > optimal_trip_option.value and total_travel_time <= detour_factor * shortest_travel_time) or (
                    value == optimal_trip_option.value and mt_travel_time < optimal_trip_option.mt_cost)):
                    optimal_trip_option = TripOption(
                        value, j, i, first_mile_travel_time, last_mile_travel_time, mt_travel_time
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


if __name__ == "__main__":
    f1 = open("manhattan_dist_1.txt", "r")
    f2 = open("manhattan_dist_2.txt", "r")
    f3 = open("manhattan_dist_3.txt", "r")
    distances = np.loadtxt(f1) + np.loadtxt(f2) + np.loadtxt(f3)

    n = 400
    potential_stops = [i for i in range(4580)]
    stops = []
    for i in range(n):
        stop_index = random.randint(0, 4580 - i - 1)
        new_stop = potential_stops.pop(stop_index)
        stops.append(new_stop)

    line_inst = line_instance(nb_lines=10, nb_pass=10, B=1, cost=1, max_length=50, min_length=6, proba=0.1, capacity=30)

    G = ox.graph_from_place('Manhattan, New York, New York, USA', network_type='drive')
    node_id = list(G.nodes())
    # impute speed on all edges missing data
    G = ox.add_edge_speeds(G)
    # calculate travel time (seconds) for all edges
    G = ox.add_edge_travel_times(G)

    f1 = open("bus_stop_number2.txt", "r")
    stops_arr = np.loadtxt(f1)
    stops_floats = stops_arr.tolist()
    stops = [int(stops_floats[i]) for i in range(len(stops_floats))]

    node_id = list(G.nodes())

    dic_id_to_index = {}  # get node number (from 0 t0 4579) from node osmid
    for i in range(len(node_id)):
        dic_id_to_index[node_id[i]] = i

    nb_lines = 1000
    min_length = 10
    max_length = 30
    min_start_end_distance = 200
    detour_skeleton = 2

    all_lines, all_routes = line_inst.generate_lines_skeleton_manhattan(
        nb_lines, stops, min_length, max_length, distances, min_start_end_distance, detour_skeleton, G
    )
    print('nb_lines_final', len(all_lines))

    all_lines_nodes = [[dic_id_to_index[all_lines[i][j]] for j in range(len(all_lines[i]))] for i in
        range(len(all_lines))]

    print('-----------')

    with open('all_lines_1000_c5.txt', 'wb') as f:
        for i in range(len(all_lines)):
            mat = np.matrix(all_lines[i])
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')

    with open('all_lines_nodes_1000_c5.txt', 'wb') as f:
        for i in range(len(all_lines_nodes)):
            mat = np.matrix(all_lines_nodes[i])
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')

    with open('all_routes_1000_c5.txt', 'wb') as f:
        for i in range(len(all_routes)):
            mat = np.matrix(all_routes[i])
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
