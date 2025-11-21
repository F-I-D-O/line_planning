import pandas as pd
from pathlib import Path

from darpinstances.solution_checker import load_data
from darpinstances.instance_objects import ActionType


def analyze_sharability(instance_path: Path):
    """
    Analyze sharability statistics for a DARP instance.
    
    Args:
        instance_path: Full path to the instance results folder (e.g., path to the folder containing config.yaml-solution.json)
    
    Returns:
        pd.DataFrame: Location statistics dataframe with columns:
            - vehicle_end_counts
            - vehicle_start_counts
            - request_origin_counts
            - request_destination_counts
            - chance_of_pickup_inflow
            - chance_of_dropoff_outflow
    """
    # Load instance and solution data
    instance, solution = load_data(instance_path / "config.yaml-solution.json", None)
    
    # Process solution into action and plan dataframes
    action_list = []
    plan_list = []
    for plan in solution.vehicle_plans:
        for index, action_data in enumerate(plan.actions):
            action = action_data.action
            if index == 0:
                plan_departure_location = action.node.get_idx()
            action_list.append([
                    plan.vehicle.index,
                    index,
                    action.action_type,
                    action_data.arrival_time,
                    action_data.departure_time,
                    action.node.get_idx()
            ])
            if index == len(plan.actions) - 1:
                plan_arrival_location = action.node.get_idx()
                plan_list.append([
                    plan.vehicle.index,
                    plan.departure_time,
                    plan.arrival_time,
                    plan_departure_location,
                    plan_arrival_location,
                    plan.cost
                ])
    
    action_df = pd.DataFrame(action_list, columns=['vehicle', 'action_index', 'action_type', 'arrival_time', 'departure_time', 'position'])
    action_df['action_type'] = pd.Categorical(action_df['action_type'])
    action_df['action_type'] = action_df.action_type.map({ActionType.PICKUP: 'pickup', ActionType.DROP_OFF: 'dropoff'})
    action_df.set_index(['vehicle', 'action_index'], inplace=True)
    
    plan_df = pd.DataFrame(plan_list, columns=['vehicle', 'departure_time', 'arrival_time', 'departure_location', 'arrival_location', 'cost'])
    plan_df.set_index('vehicle', inplace=True)
    
    # Calculate location statistics
    vehicle_end_counts = plan_df.groupby('arrival_location').size()
    vehicle_start_counts = plan_df.groupby('departure_location').size()
    
    request_origin_counts = action_df[action_df['action_type'] == 'pickup'].groupby('position').size()
    request_destination_counts = action_df[action_df['action_type'] == 'dropoff'].groupby('position').size()
    
    # Combine statistics
    location_statistics = pd.concat([vehicle_end_counts, vehicle_start_counts, request_origin_counts, request_destination_counts], axis=1)
    location_statistics.columns = ['vehicle_end_counts', 'vehicle_start_counts', 'request_origin_counts', 'request_destination_counts']
    location_statistics.index.name = 'location'
    location_statistics.fillna(0, inplace=True)
    
    # Calculate additional metrics
    location_statistics['chance_of_pickup_inflow'] = location_statistics['vehicle_start_counts'] / location_statistics['request_origin_counts']
    location_statistics['chance_of_dropoff_outflow'] = location_statistics['vehicle_end_counts'] / location_statistics['request_destination_counts']
    
    return location_statistics

