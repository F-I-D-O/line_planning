import random
import logging
import pandas as pd
import networkx as nx
import osmnx as ox

from pathlib import Path
from copy import deepcopy

from darpinstances.instance import MatrixTravelTimeProvider
from tqdm import tqdm

# default configuration (used only when running this module as a script)
# area_path = Path(r"C:\Google Drive AIC\My Drive\AIC Experiment Data\DARP\Instances\Manhattan")
DEFAULT_AREA_PATH = Path(r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning\Instances\Chyse")
DEFAULT_NUMBER_OF_STOPS = 10
DEFAULT_NB_LINES = 30
DEFAULT_MIN_LENGTH = 2
DEFAULT_MAX_LENGTH = 10
DEFAULT_MIN_START_END_DISTANCE = 20
DEFAULT_DETOUR_SKELETON = 2
# DEFAULT_AREA_NAME = 'Manhattan, New York, New York, USA'
DEFAULT_AREA_NAME = None


def load_graph(edges_path: Path):
    logging.info("Loading graph from %s", edges_path)
    edge_df = pd.read_csv(edges_path, delimiter='\t')
    DEFAULT_SPEED = 14 # 50 km/h in meters per second
    if 'speed' in edge_df.columns:
        edge_df['travel_time'] = edge_df['length'] / edge_df['speed']
    else:
        edge_df['travel_time'] = edge_df['length'] / DEFAULT_SPEED

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
        stops: list[int],
        travel_time_provider: MatrixTravelTimeProvider,
        G,
        min_start_end_distance,
        detour_skeleton,
        min_length,
        max_length
    ):
        self.stops: list[int] = stops
        self.stops_set: set[int] = set(stops)
        self.travel_time_provider: MatrixTravelTimeProvider = travel_time_provider
        self.G = G
        self.min_start_end_distance = min_start_end_distance
        self.detour_skeleton = detour_skeleton
        self.min_length = min_length
        self.max_length = max_length

    @profile
    def shortest_path_nodes(self, orig_index, dest_index):
        orig = list(self.G)[orig_index]
        dest = list(self.G)[dest_index]
        route = nx.shortest_path(self.G, orig, dest, weight='travel_time')
        # print(route)
        route_within_stops = []
        for node in route:
            if node in self.stops_set:
                route_within_stops.append(node)

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

                if iter == 10:
                    break

                try:
                    length = random.randint(self.min_length, self.max_length)
                    new_line, route = self.generate_new_line_skeleton_manhattan(length)
                    all_lines.append(new_line)
                    all_routes.append(route)
                    progress_bar.update(1)
                except:
                    logging.warning('line_construction_failed')
        return all_lines, all_routes
        

def generate_candidate_lines(
    area_path: Path,
    output_path: Path,
    number_of_stops: int = DEFAULT_NUMBER_OF_STOPS,
    nb_lines: int = DEFAULT_NB_LINES,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_start_end_distance: int = DEFAULT_MIN_START_END_DISTANCE,
    detour_skeleton: int = DEFAULT_DETOUR_SKELETON,
    area_name: str | None = DEFAULT_AREA_NAME,
) -> None:
    """
    Generate candidate lines for a given instance directory.

    The instance directory must contain:
    - a distance matrix file named one of: dm.h5, dm.hdf5, dm.csv, dm.dm
    - a road network at map/edges.csv, or alternatively provide area_name to download from OSM.
    """
    lines_path = output_path / "lines.txt"

    dm_candidates = [
        area_path / "dm.h5",
        area_path / "dm.hdf5",
        area_path / "dm.csv"
    ]
    dm_path = next((p for p in dm_candidates if p.exists()), None)
    if dm_path is None:
        raise FileNotFoundError(
            "Distance matrix file does not exist. Expected one of: %s in %s"
            % ([p.name for p in dm_candidates], area_path)
        )

    travel_time_provider = MatrixTravelTimeProvider.read_from_file(dm_path)

    logging.info("Randomly selecting stops")
    potential_stops = [i for i in range(travel_time_provider.get_node_count())]
    stops: list[int] = []
    for _ in range(number_of_stops):
        stop_index = random.randint(0, len(potential_stops) - 1)
        new_stop = potential_stops.pop(stop_index)
        stops.append(new_stop)

    edge_path = area_path / "map/edges.csv"
    if edge_path.exists():
        G = load_graph(edge_path)
    elif area_name is not None:
        G = get_graph_from_openstreetmap(area_name)
    else:
        raise ValueError("Either the edge path or the area name must be specified.")

    line_generator = CandidateLineGenerator(
        stops,
        travel_time_provider,
        G,
        min_start_end_distance,
        detour_skeleton,
        min_length,
        max_length,
    )

    logging.info("Generating candidate lines")
    all_lines, _ = line_generator.generate_lines_skeleton_manhattan(nb_lines)

    with open(lines_path, "w") as f:
        logging.info("Exporting candidate lines to %s", lines_path)
        for line in all_lines:
            f.write(",".join([str(i) for i in line]) + "\n")


if __name__ == "__main__":
    generate_candidate_lines(
        area_path=DEFAULT_AREA_PATH,
        number_of_stops=DEFAULT_NUMBER_OF_STOPS,
        nb_lines=DEFAULT_NB_LINES,
        min_length=DEFAULT_MIN_LENGTH,
        max_length=DEFAULT_MAX_LENGTH,
        min_start_end_distance=DEFAULT_MIN_START_END_DISTANCE,
        detour_skeleton=DEFAULT_DETOUR_SKELETON,
        area_name=DEFAULT_AREA_NAME,
    )
