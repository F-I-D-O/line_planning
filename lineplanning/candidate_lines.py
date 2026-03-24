import random
import logging
import pandas as pd
import networkx as nx
import osmnx as ox

from pathlib import Path
from copy import deepcopy
from typing import Any

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


def load_graph(edges_path: Path) -> nx.DiGraph:
    """
    Build a directed graph whose node labels are exactly the ``u`` / ``v`` values
    from the edge file. These must match distance-matrix row/column indices
    (0 .. n-1 or the same id space as ``MatrixTravelTimeProvider``); otherwise
    line generation and travel-time lookups are inconsistent.
    """
    logging.info("Loading graph from %s", edges_path)
    edge_df = pd.read_csv(edges_path, delimiter='\t')
    # Normalize endpoints to int labels so they match DM indices (CSV may parse as float).
    edge_df["u"] = pd.to_numeric(edge_df["u"], errors="raise").astype(int)
    edge_df["v"] = pd.to_numeric(edge_df["v"], errors="raise").astype(int)
    DEFAULT_SPEED = 14 # 50 km/h in meters per second
    if 'speed' in edge_df.columns:
        edge_df['travel_time'] = edge_df['length'] / edge_df['speed']
    else:
        edge_df['travel_time'] = edge_df['length'] / DEFAULT_SPEED

    return nx.from_pandas_edgelist(edge_df, source='u', target='v', edge_attr='travel_time', create_using=nx.DiGraph())


def full_route_for_stop_sequence(
    G: nx.DiGraph, stops: list[int], weight: str = "travel_time"
) -> list:
    """
    Concatenate shortest paths between consecutive stops (same weight as line generation).
    Matches the final stop list written to lines.txt after shortening.
    """
    if len(stops) == 0:
        return []
    if len(stops) == 1:
        return [stops[0]]
    parts: list = []
    for i in range(len(stops) - 1):
        segment = nx.shortest_path(G, stops[i], stops[i + 1], weight=weight)
        if i == 0:
            parts.extend(segment)
        else:
            parts.extend(segment[1:])
    return parts


def load_node_xy_from_map(edge_path: Path) -> dict[int, tuple[float, float]] | None:
    """
    Load node coordinates from map/nodes.csv (tab-separated, columns id, x, y).
    x/y are longitude/latitude in WGS84 (EPSG:4326), matching edges.csv u/v ids.
    """
    nodes_path = edge_path.parent / "nodes.csv"
    if not nodes_path.exists():
        logging.info("Geo export skipped: nodes file not found at %s", nodes_path)
        return None
    node_df = pd.read_csv(nodes_path, delimiter="\t")
    required = {"id", "x", "y"}
    if not required.issubset(node_df.columns):
        logging.warning(
            "Geo export skipped: %s must contain columns %s (found %s)",
            nodes_path,
            sorted(required),
            list(node_df.columns),
        )
        return None
    out: dict[int, tuple[float, float]] = {}
    for _, row in node_df.iterrows():
        out[int(row["id"])] = (float(row["x"]), float(row["y"]))
    return out


def write_candidate_lines_gpkg(
    stop_sequences: list[list[int]],
    G: nx.DiGraph,
    node_xy: dict[int, tuple[float, float]],
    gpkg_path: Path,
) -> None:
    """Write routes (LineString) and stops (Point) layers to a GeoPackage (EPSG:4326)."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    crs = "EPSG:4326"
    route_records: list[dict[str, Any]] = []
    stop_records: list[dict[str, Any]] = []

    for line_id, stops in enumerate(stop_sequences):
        route_nodes = full_route_for_stop_sequence(G, stops)
        coords: list[tuple[float, float]] = []
        skip_route = False
        for nid in route_nodes:
            if int(nid) not in node_xy:
                logging.warning(
                    "Node %s not in nodes.csv; skipping route geometry for line_id %s",
                    nid,
                    line_id,
                )
                skip_route = True
                break
            xy = node_xy[int(nid)]
            coords.append(xy)
        if not skip_route and coords:
            if len(coords) == 1:
                geom: LineString = LineString([coords[0], coords[0]])
            else:
                geom = LineString(coords)
            route_records.append(
                {
                    "line_id": line_id,
                    "n_stops": len(stops),
                    "n_vertices": len(route_nodes),
                    "geometry": geom,
                }
            )

        for stop_seq, node_id in enumerate(stops):
            if node_id not in node_xy:
                logging.warning(
                    "Node %s not in nodes.csv; skipping stop point line_id %s stop_seq %s",
                    node_id,
                    line_id,
                    stop_seq,
                )
                continue
            x, y = node_xy[node_id]
            stop_records.append(
                {
                    "line_id": line_id,
                    "stop_seq": stop_seq,
                    "node_id": node_id,
                    "geometry": Point(x, y),
                }
            )

    if not route_records and not stop_records:
        logging.warning("Geo export skipped: no geometries could be built from nodes.csv")
        return

    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        gpkg_path.unlink()

    if route_records:
        gpd.GeoDataFrame(route_records, crs=crs).to_file(
            gpkg_path, layer="routes", driver="GPKG"
        )
    if stop_records:
        mode = "a" if gpkg_path.exists() else "w"
        gpd.GeoDataFrame(stop_records, crs=crs).to_file(
            gpkg_path, layer="stops", driver="GPKG", mode=mode
        )

    logging.info("Wrote candidate line geometries to %s", gpkg_path)


def get_graph_from_openstreetmap(area_name: str):
    G = ox.graph_from_place(area_name, network_type='drive')
    # impute speed on all edges missing data
    G = ox.add_edge_speeds(G)
    # calculate travel time (seconds) for all edges
    G = ox.add_edge_travel_times(G)

    # Relabel nodes to 0..n-1; the distance matrix must use the same indices.
    G = nx.convert_node_labels_to_integers(G)

    return G


class CandidateLineGenerator:
    def __init__(
        self,
        stops: list[int],
        travel_time_provider: MatrixTravelTimeProvider,
        G: nx.DiGraph,
        min_start_end_distance,
        detour_skeleton,
        min_length,
        max_length
    ):
        self.stops: list[int] = stops
        self.stops_set: set[int] = set(stops)
        self.travel_time_provider: MatrixTravelTimeProvider = travel_time_provider
        self.G: nx.DiGraph = G
        self.min_start_end_distance = min_start_end_distance
        self.detour_skeleton = detour_skeleton
        self.min_length = min_length
        self.max_length = max_length

    def compute_route_between_stops(self, orig_node_label: int, dest_node_label: int):
        """
        Shortest path on the road graph between two stops, using **graph node labels**
        (same ids as DM / ``u``–``v`` in ``edges.csv``), not positions in ``list(G)``.
        """
        route = nx.shortest_path(
            self.G, orig_node_label, dest_node_label, weight="travel_time"
        )
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

        route_within_stops_1, route1 = self.compute_route_between_stops(start, inter)
        route_within_stops_2, route2 = self.compute_route_between_stops(inter, inter_2)
        route_within_stops_3, route3 = self.compute_route_between_stops(inter_2, end)

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
    export_geopackage: bool = True,
    geopackage_path: Path | None = None,
) -> None:
    """
    Generate candidate lines for a given instance directory.

    The instance directory must contain:
    - a distance matrix file named one of: dm.h5, dm.hdf5, dm.csv, dm.dm
    - a road network at map/edges.csv, or alternatively provide area_name to download from OSM.

    Writes lines.txt under output_path. When the graph is loaded from map/edges.csv and
    map/nodes.csv exists (id, x, y tab-separated; WGS84 lon/lat), also writes
    candidate_lines.gpkg (routes + stops layers) for QGIS unless disabled or geopandas
    **Graph / DM contract:** ``edges.csv`` endpoints ``u`` and ``v`` are node labels and must
    match distance-matrix indices (same integer id space). Shortest paths use these labels
    directly—never list positions in ``G``.
    is not installed.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

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
    node_xy: dict[int, tuple[float, float]] | None = None
    if edge_path.exists():
        G: nx.DiGraph = load_graph(edge_path)
        node_xy = load_node_xy_from_map(edge_path)
    elif area_name is not None:
        G: nx.DiGraph = get_graph_from_openstreetmap(area_name)
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
    lines_path = output_path / "lines.txt"

    with open(lines_path, "w") as f:
        logging.info("Exporting candidate lines to %s", lines_path)
        for line in all_lines:
            f.write(",".join([str(i) for i in line]) + "\n")

    if (
        export_geopackage
        and edge_path.exists()
        and node_xy
    ):
        gpkg_out = (
            Path(geopackage_path)
            if geopackage_path is not None
            else output_path / "candidate_lines.gpkg"
        )
        try:
            write_candidate_lines_gpkg(all_lines, G, node_xy, gpkg_out)
        except ImportError:
            logging.warning(
                "Geo export skipped: geopandas is not installed "
                "(install with: pip install lineplanning[viz] or pip install geopandas)"
            )
    elif export_geopackage and not edge_path.exists():
        logging.info("Geo export skipped: graph not loaded from map/edges.csv")
    elif export_geopackage and edge_path.exists() and not node_xy:
        pass


if __name__ == "__main__":
    generate_candidate_lines(
        area_path=DEFAULT_AREA_PATH,
        output_path=DEFAULT_AREA_PATH,
        number_of_stops=DEFAULT_NUMBER_OF_STOPS,
        nb_lines=DEFAULT_NB_LINES,
        min_length=DEFAULT_MIN_LENGTH,
        max_length=DEFAULT_MAX_LENGTH,
        min_start_end_distance=DEFAULT_MIN_START_END_DISTANCE,
        detour_skeleton=DEFAULT_DETOUR_SKELETON,
        area_name=DEFAULT_AREA_NAME,
    )
