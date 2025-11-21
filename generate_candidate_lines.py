import random
import pandas as pd
import networkx as nx
import osmnx as ox


from pathlib import Path

from darpinstances.instance import MatrixTravelTimeProvider

# configuration
area_path = Path(r"C:\Google Drive AIC\My Drive\AIC Experiment Data\DARP\Instances\Manhattan")
number_of_stops = 400
nb_lines = 1000
min_length = 10
max_length = 30
min_start_end_distance = 200
detour_skeleton = 2
# area_name = 'Manhattan, New York, New York, USA'
area_name = None


def load_graph(edges_path: Path):
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


travel_time_provider = MatrixTravelTimeProvider.from_hdf(area_path / 'dm.h5')

# randomly selecting nodes for stops
potential_stops = [i for i in range(travel_time_provider.get_node_count())]
stops = []
for i in range(number_of_stops):
    stop_index = random.randint(0, len(potential_stops) - 1)
    new_stop = potential_stops.pop(stop_index)
    stops.append(new_stop)

# get the road network graph
edge_path = area_path / 'edges.csv'
if edge_path.exists():
    G = load_graph(edge_path)
elif area_name is not None:
    G = get_graph_from_openstreetmap(area_name)
else:
    raise ValueError('Either the edge path or the area name must be specified.')


line_inst = line_instance(nb_lines=10, nb_pass=10, B=1, cost=1, max_length=50, min_length=6, proba=0.1, capacity=30)

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