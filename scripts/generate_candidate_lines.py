import random
import logging
import pandas as pd
import networkx as nx
import osmnx as ox

from pathlib import Path
from copy import deepcopy

from darpinstances.instance import MatrixTravelTimeProvider
from tqdm import tqdm

# configuration
area_path = Path(r"C:\Google Drive AIC\My Drive\AIC Experiment Data\DARP\Instances\Manhattan")
lines_path = area_path / 'lines.txt'
number_of_stops = 400
nb_lines = 1000
min_length = 10
max_length = 30
min_start_end_distance = 200
detour_skeleton = 2
# area_name = 'Manhattan, New York, New York, USA'
area_name = None


def load_graph(edges_path: Path):
    logging.info("Loading graph from %s", edges_path)
    edge_df = pd.read_csv(edges_path, delimiter='\t')
    edge_df['travel_time'] = edge_df['length'] / edge_df['speed']

    return nx.from_pandas_edgelist(edge_df, source='u', target='v', edge_attr='travel_time', create_using=nx.DiGraph())


def get_graph_from_openstreetmap(area_name: str):
    G = ox.graph_from_place(area_name, network_type='drive')
    # impute speed on all edges missing data
    G = ox.add_edge_speeds(G)
    # calculate travel time (seconds) for all edges
    G = ox.add_edge_travel_times(G)

    # relabel nodes to index from 0 to n-1
    G = nx.convert_node_labels_to_integers(G)

    return G


class CandidateLineGenerator:
    def __init__(
        self,
        stops,
        travel_time_provider: MatrixTravelTimeProvider,
        G,
        min_start_end_distance,
        detour_skeleton,
        min_length,
        max_length
    ):
        self.stops = stops
        self.travel_time_provider: MatrixTravelTimeProvider = travel_time_provider
        self.G = G
        self.min_start_end_distance = min_start_end_distance
        self.detour_skeleton = detour_skeleton
        self.min_length = min_length
        self.max_length = max_length

    def shortest_path_nodes(self, orig_index, dest_index):
        orig = list(self.G)[orig_index]
        dest = list(self.G)[dest_index]
        route = nx.shortest_path(self.G, orig, dest, weight='travel_time')
        # print(route)
        nb = 0
        route_within_stops = []
        for i in range(len(route)):
            for j in range(len(self.stops)):
                if route[i] == list(self.G)[self.stops[j]]:
                    nb += 1
                    route_within_stops.append(route[i])
                    break

        return route_within_stops, route

    def generate_new_line_skeleton_manhattan(self, length):
        # TODO: pick only nodes which are in shortest_paths inter remaining_stops
        remaining_stops = deepcopy(self.stops)
        n = len(remaining_stops)
        min_distance = self.min_start_end_distance

        # distances contains for i,j the length of a shortest path between i,j
        # Chose randomly the initial node and the last node of the line
        start_index = random.randint(0, n - 1)
        start = remaining_stops.pop(start_index)

        i = 0
        while i <= 1000:
            i += 1
            end_index = random.randint(0, n - 2)
            end = remaining_stops[end_index]
            if self.travel_time_provider.get_travel_time(start, end) >= min_distance:
                break

        end = remaining_stops.pop(end_index)

        i = 0
        while i <= 1000:
            i += 1
            inter_index = random.randint(0, n - 3)
            inter = remaining_stops[inter_index]
            if self.travel_time_provider.get_travel_time(start, inter) + self.travel_time_provider.get_travel_time(end, inter) <= \
                self.travel_time_provider.get_travel_time(start, end) * self.detour_skeleton:
                break
        inter = remaining_stops.pop(inter_index)

        i = 0
        while i <= 1000:
            i += 1
            inter_index_2 = random.randint(0, n - 4)
            inter_2 = remaining_stops[inter_index_2]
            if self.travel_time_provider.get_travel_time(inter_2, inter) + self.travel_time_provider.get_travel_time(end, inter_2) <= \
                self.travel_time_provider.get_travel_time(inter, end) * self.detour_skeleton:
                break
        inter_2 = remaining_stops.pop(inter_index_2)

        route_within_stops_1, route1 = self.shortest_path_nodes(start, inter)
        route_within_stops_2, route2 = self.shortest_path_nodes(inter, inter_2)
        route_within_stops_3, route3 = self.shortest_path_nodes(inter_2, end)

        line = route_within_stops_1 + route_within_stops_2[1:] + route_within_stops_3[1:]
        route = route1 + route2[1:] + route3[1:]

        while len(line) > length:
            index = random.randint(1, len(line) - 2)
            line.pop(index)
        return line, route

    def generate_lines_skeleton_manhattan(self, nb_lines):
        all_lines = []
        all_routes = []
        iter = 0
        with tqdm(total=nb_lines, desc='Generating candidate lines') as progress_bar:
            while len(all_lines) < nb_lines and iter < 2 * nb_lines:
                iter += 1
                try:
                    length = random.randint(self.min_length, self.max_length)
                    new_line, route = self.generate_new_line_skeleton_manhattan(length)
                    all_lines.append(new_line)
                    all_routes.append(route)
                    progress_bar.update(1)
                except:
                    logging.warning('line_construction_failed')
        return all_lines, all_routes


travel_time_provider = MatrixTravelTimeProvider.from_hdf(area_path / 'dm.h5')

# randomly selecting nodes for stops
logging.info("Randomly selecting stops")
potential_stops = [i for i in range(travel_time_provider.get_node_count())]
stops = []
for i in range(number_of_stops):
    stop_index = random.randint(0, len(potential_stops) - 1)
    new_stop = potential_stops.pop(stop_index)
    stops.append(new_stop)

# get the road network graph
edge_path = area_path / 'map/edges.csv'
if edge_path.exists():
    G = load_graph(edge_path)
elif area_name is not None:
    G = get_graph_from_openstreetmap(area_name)
else:
    raise ValueError('Either the edge path or the area name must be specified.')

line_generator = CandidateLineGenerator(
    stops, travel_time_provider, G, min_start_end_distance, detour_skeleton, min_length, max_length
)

logging.info("Generating candidate lines")
all_lines, all_routes = line_generator.generate_lines_skeleton_manhattan(nb_lines)


with open(lines_path, 'w') as f:
    logging.info("Exporting candidate lines to %s", lines_path)
    for line in all_lines:
        f.write(','.join([str(i) for i in line]) + '\n')
