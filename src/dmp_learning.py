import numpy as np
import pandas as pd
import argparse
import sys
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from movement_primitives.dmp import CartesianDMP

def parse_time_to_seconds(time_str):
    """Convert time string (MM:SS.ss) to seconds."""
    minutes, seconds = time_str.split(':')
    return float(minutes) * 60 + float(seconds)

def load_trajectory_data(csv_path, start_time, end_time):
    """Load and filter trajectory data for a specific time range."""
    # Load the trajectory data
    data = pd.read_csv(csv_path)
    
    # Convert timestamps to seconds from the start
    data['timestamp'] = data['timestamp'] - data['timestamp'].iloc[0]
    
    # Filter data for the action's time range
    mask = (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
    return data[mask]

def learn_dmp_from_trajectory(trajectory_data, weights_per_dim=200, dt=0.01, int_dt=0.001):

    position = trajectory_data[['x', 'y', 'z']].values
    orientation = trajectory_data[['qx', 'qy', 'qz', 'qw']].values
    trajectory = np.hstack((position, orientation))
    
    time = trajectory_data['timestamp'].values - trajectory_data['timestamp'].iloc[0]
    execution_time = time[-1] if len(time) > 0 else 0
    
    if execution_time <= 0:
        raise ValueError("Invalid execution time for trajectory")
    
    dmp = CartesianDMP(
        n_weights_per_dim=weights_per_dim,
        dt=dt,
        execution_time=execution_time,
        int_dt=int_dt
    )
    dmp.imitate(time, trajectory)
    
    return dmp

def process_json_file(json_path):
    """Process a JSON file and generate DMPs for each action."""
   
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    json_dir = os.path.dirname(os.path.abspath(json_path))
    
    demo_data = data.get('demonstration', {})
    trajectory_file = demo_data.get('trajectories')
    if not trajectory_file:
        raise ValueError("No trajectory file specified in JSON")
    
    if not os.path.isabs(trajectory_file):
        trajectory_file = os.path.join(json_dir, trajectory_file)
    
    # Process each action
    actions = demo_data.get('actions', [])
    if not actions:
        print("No actions found in the JSON file")
        return
    
    for i, action in enumerate(actions, 1):
        try:
            start_time = parse_time_to_seconds(action['start_time'].replace(' ', '').replace('s', ''))
            end_time = parse_time_to_seconds(action['end_time'].replace(' ', '').replace('s', ''))
            
            trajectory_data = load_trajectory_data(trajectory_file, start_time, end_time)
            
            if len(trajectory_data) < 2:
                print(f"Skipping action {i}: Not enough data points")
                continue
            
            # Create dmps directory if it doesn't exist
            dmps_dir = os.path.join(json_dir, 'dmps')
            os.makedirs(dmps_dir, exist_ok=True)
            
            # Generate a clean filename from the action description
            # Keep original capitalization, replace spaces with underscores, and remove periods
            clean_desc = action['description'].replace(' ', '_').replace('.', '')
            clean_desc = "".join(c if c.isalnum() or c == '_' else '' for c in clean_desc)
            
            output_filename = f"{clean_desc}.pkl"
            output_path = os.path.join(dmps_dir, output_filename)
            
            dmp = learn_dmp_from_trajectory(trajectory_data)
            
            with open(output_path, 'wb') as f:
                pickle.dump(dmp, f)
            
            print(f"Saved DMP for action {i}: {action['description']} to {output_path}")
            
        except Exception as e:
            print(f"Error processing action {i} ({action.get('description', 'no description')}): {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process JSON file with action demonstrations and generate DMPs')
    parser.add_argument('json_file', help='Path to JSON file containing action demonstrations')
    args = parser.parse_args()
    
    process_json_file(args.json_file)

if __name__ == "__main__":
    main()


