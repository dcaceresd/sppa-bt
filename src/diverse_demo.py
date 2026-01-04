from PIL import Image
import copy
import os
import logging
import sys
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from movement_primitives.dmp import CartesianDMP

from src.scene_graph import SceneGraph
from src.build_bt import build_behavior_tree_recursive, save_tree_to_file

from src.action_fusion import Action, ActionFuser, ActionSegment
from src.sam_lang import GeometricRelationAnalyzer

import json
import cv2
import pandas as pd

from pathlib import Path


def save_all_demonstrations_to_file(all_demonstrations, filename):
    """
    Save the list of all demonstrations (actions) to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(all_demonstrations, f)
    print(f"Saved all demonstrations to {filename}")


def load_all_demonstrations_from_file(filename):
    """
    Load the list of all demonstrations (actions) from a file.
    Args:
        filename (str): Path to the file to load.
    Returns:
        list: The loaded demonstrations/actions.
    """
    with open(filename, 'rb') as f:
        all_demonstrations = pickle.load(f)
    print(f"Loaded all demonstrations from {filename}")
    return all_demonstrations


def parse_timestamp(timestamp):
    """Parse timestamp string in format 'MM:SS.mmm' or 'HH:MM:SS.mmm' to seconds."""
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
        
    if not isinstance(timestamp, str):
        raise ValueError(f"Timestamp must be string, int, or float, got {type(timestamp)}: {timestamp}")
        
    try:
        # Handle format like "00:15.20"
        if timestamp.count(':') == 1:  # MM:SS.mmm
            minutes, seconds = timestamp.split(':')
            return float(minutes) * 60 + float(seconds)
        # Handle format like "00:00:15.200"
        elif timestamp.count(':') == 2:  # HH:MM:SS.mmm
            parts = timestamp.split(':')
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(timestamp)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS.mmm format."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    return f"{int(m):02d}:{s:06.3f}"


# DMP functions
def learn_dmp_from_multiple_demos(demonstrations):
    """
    Learn a single DMP from multiple demonstration trajectories.
    """
    max_execution_time = max([demo['timestamp'].iloc[-1] - demo['timestamp'].iloc[0] for demo in demonstrations])
    dt = 0.01
    
    dmp = CartesianDMP(
        n_weights_per_dim=200,  
        dt=dt,
        execution_time=max_execution_time,
        int_dt=0.001
    )
    
    for demo in demonstrations:
        time = demo['timestamp'].values - demo['timestamp'].iloc[0]
        position = demo[['x', 'y', 'z']].values
        orientation = demo[['qx', 'qy', 'qz', 'qw']].values
        traj = np.hstack((position, orientation))
        
        dmp.imitate(time, traj)
    
    return dmp

def _dtw_distance(traj1, traj2):
    """Calculate Dynamic Time Warping distance between two trajectories."""
    n, m = len(traj1), len(traj2)
    cost = np.full((n+1, m+1), np.inf)
    cost[0, 0] = 0
    
    pos1 = traj1[['x', 'y', 'z']].values
    pos2 = traj2[['x', 'y', 'z']].values
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            diff = np.linalg.norm(pos1[i-1] - pos2[j-1])
            cost[i, j] = diff + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    
    return cost[n, m] / max(n, m)  # Normalize by the longer trajectory length

def learn_and_save_dmps(fused_actions, base_dir, reference_demo=None):
    """
    Group similar segments by group_label, gripper status, and DTW similarity,
    then learn DMPs for each group and save them in a 'dmps' folder.

    """
    dmps_dir = os.path.join(base_dir, 'dmps')
    os.makedirs(dmps_dir, exist_ok=True)
    
    all_segments = defaultdict(list)
    
    for demo in fused_actions:  
        for action in demo:  
            if not hasattr(action, 'segments') or not action.segments:
                print(f"Warning: Action {action.label} has no segments, skipping...")
                continue
            
            traj_df = pd.DataFrame(
                action.trajectory,
                columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
            )
            
            for segment in action.segments:
                start_time = parse_timestamp(segment.start_time if hasattr(segment, 'start_time') else segment.start_time_sec)
                end_time = parse_timestamp(segment.end_time if hasattr(segment, 'end_time') else segment.end_time_sec)
                
                segment_data = traj_df[
                    (traj_df['timestamp'] >= start_time) & 
                    (traj_df['timestamp'] <= end_time)
                ][['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values
                
                if not segment_data.size:
                    continue
                    
                segment_data = pd.DataFrame(segment_data, columns=["x", "y", "z", "qx", "qy", "qz", "qw"])
                segment_data['timestamp'] = np.arange(len(segment_data))
                segment_data['timestamp'] -= segment_data['timestamp'].iloc[0]
                
                # Add segment data to the appropriate group
                all_segments[action.group_label].append({
                    'data': segment_data,
                    'action_label': action.label,
                    'segment': segment,
                    'source_action': action,
                    'gripper_state': segment.gripper_state
                })
    
    if not fused_actions:
        print("No fused actions available")
        return
    
    print(f"Using {len(fused_actions)} fused actions")
    
    if reference_demo is None:
        print("No reference demo provided. Learning DMPs for each action individually.")
        for group_label, group_segments in all_segments.items():
            for i, segment_info in enumerate(group_segments):
                try:
                    dmp = learn_dmp_from_multiple_demos([segment_info['data']])
                    
                    segment_label = f"segment_{i:03d}"
                    safe_label = "".join([c if c.isalnum() else "_" for c in str(segment_info['action_label'])])
                    safe_group = "".join([c if c.isalnum() else "_" for c in str(group_label)])
                    gripper_str = "open" if segment_info['gripper_state'] else "closed"
                    
                    filename = f"{safe_label}-{safe_group}-{gripper_str}-{segment_label}.pkl"
                    filepath = os.path.join(dmps_dir, filename)
                    
                    dmp_info = {
                        'dmp': dmp,
                        'group_label': group_label,
                        'gripper_state': segment_info['gripper_state'],
                        'action_label': segment_info['action_label'],
                        'segment_label': segment_label,
                        'num_segments': 1,
                        'segment_duration_avg': len(segment_info['data']),
                        'reference_segment': f"{segment_info['action_label']}_{i}"
                    }
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(dmp_info, f)
                        
                    print(f"Saved DMP at {filepath} for individual segment")
                    
                except Exception as e:
                    print(f"Error learning DMP for segment {i} in group {group_label}: {str(e)}")
        return
    
    print(f"Processing {len(reference_demo)} actions in reference demo")
    for i, ref_action in enumerate(reference_demo):
        print(f"\nAction {i + 1}/{len(reference_demo)}:")
        print(f"  Type: {type(ref_action)}")
        if hasattr(ref_action, '__dict__'):
            print("  Attributes:", [attr for attr in dir(ref_action) if not attr.startswith('_')])
        else:
            print(f"  Value: {ref_action}")
            
        if not hasattr(ref_action, 'segments') or not ref_action.segments:
            print(f"  Skipping action {getattr(ref_action, 'label', 'unknown')} - no segments")
            continue
            
        print(f"Processing reference action: {getattr(ref_action, 'label', 'unknown')} with {len(ref_action.segments)} segments")
            
        ref_traj_df = pd.DataFrame(
            ref_action.trajectory,
            columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
        )
        
        for ref_segment in ref_action.segments:
            start_time = parse_timestamp(ref_segment.start_time if hasattr(ref_segment, 'start_time') else ref_segment.start_time_sec)
            end_time = parse_timestamp(ref_segment.end_time if hasattr(ref_segment, 'end_time') else ref_segment.end_time_sec)
            
            ref_segment_data = ref_traj_df[
                (ref_traj_df['timestamp'] >= start_time) & 
                (ref_traj_df['timestamp'] <= end_time)
            ].copy()
            
            if ref_segment_data.empty:
                print("Empty reference segment data")
                continue
                
            ref_segment_data['timestamp'] -= ref_segment_data['timestamp'].iloc[0]
            
            if ref_action.group_label not in all_segments:
                print("No segments found for group key:", ref_action.group_label)
                continue
                
            print("Finding similar segments")
            similar_segments = []
            for segment_info in all_segments[ref_action.group_label]:
                if segment_info['gripper_state'] != ref_segment.gripper_state:
                    continue
                else:
                    try:
                        distance = _dtw_distance(ref_segment_data, segment_info['data'])
                        if distance == 0:
                            print(f"  Found identical segment with distance {distance:.6f}")
                        elif distance < 0.1:
                            similar_segments.append(segment_info)
                            print(f"  Added segment with distance {distance:.6f}")
                    except Exception as e:
                        print(f"Error calculating DTW distance: {str(e)}")
            
            if not similar_segments:
                print("No similar segments found, using reference segment")
                ref_segment_info = {
                    'data': ref_segment_data,
                    'action_label': ref_action.label,
                    'segment': ref_segment,
                    'source_action': ref_action,
                    'gripper_state': ref_segment.gripper_state
                }
                similar_segments = [ref_segment_info]
                print(f"Learning DMP from reference segment {ref_segment.label}")
            else:
                print(f"Found {len(similar_segments)} segments similar to reference segment {ref_segment.label}")
            
            try:
                segment_data_list = [s['data'] for s in similar_segments]
                
                dmp = learn_dmp_from_multiple_demos(segment_data_list)
                
                first_segment = similar_segments[0]
                action_label = first_segment['action_label']
                segment_label = first_segment['segment'].label.lower().replace(' ', '_').replace('.', '')
                safe_label = action_label.lower().replace(' ', '_').replace('.', '')
                safe_group = str(ref_action.group_label).lower().replace(' ', '_').replace('.', '')
                gripper_str = ref_segment.gripper_state
                
                filename = f"{safe_label}-{safe_group}-{gripper_str}-{segment_label}.pkl"
                filepath = os.path.join(dmps_dir, filename)
                
                dmp_info = {
                    'dmp': dmp,
                    'group_label': ref_action.group_label,
                    'gripper_state': ref_segment.gripper_state,
                    'action_label': ref_action.label,
                    'segment_label': segment_label,
                    'num_segments': len(similar_segments),
                    'segment_duration_avg': np.mean([len(s['data']) for s in similar_segments]),
                    'reference_segment': ref_segment.label
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(dmp_info, f)
                    
                print(f"Saved DMP at {filepath} with {len(similar_segments)} similar segments")
                
            except Exception as e:
                print(f"Error learning DMP for segment {ref_segment.label}: {str(e)}")

# BT functions
def build_behavior_tree_from_multiple_demos(all_demonstrations):
    """
    Build a behavior tree from multiple demonstrations.
    """
    reference_demo = find_reference_demonstration(all_demonstrations)
    print(f"Selected reference demo with {len(reference_demo)} actions")
    for i, action in enumerate(reference_demo):
        print(f"  Action {i+1}: {getattr(action, 'label', 'unknown')} with {len(getattr(action, 'segments', []))} segments")
    
    action_order = determine_action_order(all_demonstrations, reference_demo)
    
    return build_behavior_tree_recursive(reference_demo)

def find_reference_demonstration(all_demonstrations):
    """
    Find the demonstration with the least number of pre- and post-conditions.
    """
    min_conditions = float('inf')
    reference_demo = None
    
    for demo in all_demonstrations:
        total_conditions = sum(
            (len(pre_conditions) if isinstance(pre_conditions, (list, set)) else 1) +
            (len(post_conditions) if isinstance(post_conditions, (list, set)) else 1)
            for _, post_conditions, pre_conditions in demo
        )
        
        if total_conditions < min_conditions:
            min_conditions = total_conditions
            reference_demo = demo
    
    return reference_demo


def find_reference_demo_from_actions(actions_or_demos):
    """
    Find a reference demonstration from either:
    - A list of Action objects, or
    - A list of demonstrations (each being a list of Action objects)
    """
    if not actions_or_demos:
        return None
        
    # If the first element is an Action object, treat the input as a single demonstration
    if hasattr(actions_or_demos[0], 'trajectory'):
        print("Single demonstration.")
        reference_demo = actions_or_demos
    else:
        # Otherwise, treat as a list of demonstrations and find the one with most actions
        print("List of demonstrations.")
        reference_demo = max(actions_or_demos, key=len)
    
    print(f"Selected reference demo with {len(reference_demo)} actions")
    for i, action in enumerate(reference_demo):
        print(f"  Action {i+1}: {getattr(action, 'label', 'unknown')} with {len(getattr(action, 'segments', []))} segments")
    
    return reference_demo

def determine_action_order(all_demonstrations, reference_demo):
    """
    Determine the most common order of actions across all demonstrations.
    If no common order can be determined, use the reference demo's order.
    """
    action_sequences = []
    for demo in all_demonstrations:
        action_sequences.append([action for action, _, _ in demo])
    
    action_pairs = Counter()
    for sequence in action_sequences:
        for i in range(len(sequence) - 1):
            action_pairs[(sequence[i], sequence[i+1])] += 1
    
    ordered_actions = []
    remaining_actions = set(action for demo in all_demonstrations for action, _, _ in demo)
    
    first_actions = Counter(seq[0] for seq in action_sequences if seq)
    if first_actions:
        most_common_first = first_actions.most_common(1)[0][0]
        ordered_actions.append(most_common_first)
        remaining_actions.remove(most_common_first)
    
    while remaining_actions and len(ordered_actions) < len(remaining_actions) + 1:
        current = ordered_actions[-1]
        next_candidates = [(next_action, count) for (curr, next_action), count in action_pairs.items() 
                          if curr == current and next_action in remaining_actions]
        
        if not next_candidates:
            break
            
        next_action = max(next_candidates, key=lambda x: x[1])[0]
        ordered_actions.append(next_action)
        remaining_actions.remove(next_action)
    
    if remaining_actions:
        reference_actions = [action for action, _, _ in reference_demo]
        final_order = ordered_actions.copy()
        
        for action in reference_actions:
            if action not in final_order and action in remaining_actions:
                final_order.append(action)
                remaining_actions.remove(action)
        
        final_order.extend(remaining_actions)
        return final_order
    
    return ordered_actions

def extract_common_conditions(all_demonstrations, action_order):
    """
    Extract common pre- and post-conditions for each action across demonstrations.
    
    Args:
        all_demonstrations (list): List of demonstrations.
        action_order (list): Ordered list of action names.
    
    Returns:
        list: List of (action, common_post_conditions, common_pre_conditions) tuples.
    """
    action_conditions = defaultdict(lambda: {'pre': [], 'post': []})
    
    for demo in all_demonstrations:
        for action, post_conditions, pre_conditions in demo:
            pre_set = set(pre_conditions) if isinstance(pre_conditions, (list, set)) else {pre_conditions}
            post_set = set(post_conditions) if isinstance(post_conditions, (list, set)) else {post_conditions}
            
            action_conditions[action]['pre'].append(pre_set)
            action_conditions[action]['post'].append(post_set)
    
    common_conditions = []
    for action in action_order:
        if action in action_conditions:
            if action_conditions[action]['pre']:
                common_pre = set.intersection(*action_conditions[action]['pre'])
            else:
                common_pre = set()
                
            if action_conditions[action]['post']:
                common_post = set.intersection(*action_conditions[action]['post'])
            else:
                common_post = set()
            
            if len(common_pre) == 1:
                common_pre = list(common_pre)[0]
            elif len(common_pre) > 1:
                common_pre = list(common_pre)
                
            if len(common_post) == 1:
                common_post = list(common_post)[0]
            elif len(common_post) > 1:
                common_post = list(common_post)
                
            common_conditions.append((action, common_post, common_pre))
    
    return common_conditions

# Video functions
def save_action_frames(action, start_frame, end_frame, trajectory, base_dir):
    """Save frames and trajectory to demonstration folder structure"""
    if start_frame is None or end_frame is None:
        print("Error: Failed to extract start or end frame")
        return None
    
    demo_dir = os.path.join(base_dir, "actions")

    # Create action-specific folder name
    action_folder = action["description"].lower().replace(" ", "_").replace(".", "")
    action_path = os.path.join(demo_dir, action_folder)

    # Create directories if needed
    os.makedirs(action_path, exist_ok=True)
    # Save frames
    start_frame.save(os.path.join(action_path, "start.png"))
    end_frame.save(os.path.join(action_path, "end.png"))
    pd.DataFrame(trajectory, columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]).to_csv(
        os.path.join(action_path, "trajectory.csv"), index=False
    )
    return action_path

def extract_frame(video_path, timestamp):
    """Extract frame from video at specific timestamp"""
    print("Video path:", video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(round(timestamp * fps)) # int(timestamp * fps)
    print("Extracting frame at timestamp:", frame_num, timestamp, fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        print(f"Failed to extract frame at timestamp {timestamp}")
        return None

# Util functions
def time_to_seconds(time_str):
    """
    Convert a time string (MM:SS.ss or HH:MM:SS.ss) to total seconds (float).
    """
    parts = time_str.split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        # MM:SS.ss format
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # HH:MM:SS.ss format
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format")

if __name__ == "__main__":
    vlm = GeometricRelationAnalyzer()

    # Check if we're processing multiple demonstrations
    if len(sys.argv) < 2:
        print("Usage: python script.py <json_config_path> [<json_config_path2> ...]")
        sys.exit(1)
    
    all_actions = []
    all_fused_actions = []
    all_action_lists = []
    saved_folders = []
    
    for json_path in sys.argv[1:]:
        with open(json_path) as f:
            config = json.load(f)

        base_dir = os.path.dirname(os.path.abspath(json_path))
        print(f"Processing demonstration from: {base_dir}")
        base_path = str(Path(json_path).with_suffix(''))
        base_path = base_path.replace("_adjusted_gripper_trimmed", "")
        video_path = f"{base_path}.mp4"
        output_dir = os.path.join(base_dir, "actions")
        os.makedirs(output_dir, exist_ok=True)

        trajectory_path = os.path.join(base_dir, config["demonstration"]["trajectories"])
        trajectory_df = pd.read_csv(trajectory_path)

        gripper_path = os.path.join(base_dir, config["demonstration"]["gripper"])
        gripper_df = pd.read_csv(gripper_path)

        time_start = trajectory_df["timestamp"].min()
        gripper_df["timestamp"] = gripper_df["timestamp"] - time_start
        trajectory_df["timestamp"] = trajectory_df["timestamp"] - time_start

        actions = []
        detected_objects = config["demonstration"]["detected_objects"] if "detected_objects" in config["demonstration"] else []

        # Process each action from JSON
        saved_folders = []
        for action in config["demonstration"]["actions"]:
            try:
                start_frame = extract_frame(video_path, time_to_seconds(action["start_time"]))
                end_frame = extract_frame(video_path, time_to_seconds(action["end_time"]))
                
                mask = (trajectory_df["timestamp"] >= time_to_seconds(action["start_time"])) & \
                      (trajectory_df["timestamp"] <= time_to_seconds(action["end_time"]))
                try:
                    trajectory = trajectory_df[mask][["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]].values.tolist()
                except Exception as e:
                    logging.error(f"Error extracting trajectory segment: {e} for action: {action['description']}")

                action_path = save_action_frames(action, start_frame, end_frame, trajectory, output_dir)

                try:
                    start_relationships = vlm.analyze_image(os.path.join(action_path, "start.png"), detected_objects)
                    pre_conditions = [f"{item['object1']} {item['relation']} {item['object2']}" for item in start_relationships]
                except Exception as e:
                    logging.error(f"Error analyzing pre-conditions: {e} for action: {action['description']}")

                try:
                    end_relationships = vlm.analyze_image(os.path.join(action_path, "end.png"), detected_objects)
                    post_conditions = [f"{item['object1']} {item['relation']} {item['object2']}" for item in end_relationships]
                except Exception as e:
                    logging.error(f"Error analyzing post-conditions: {e} for action: {action['description']}")

                try:
                    segments = []
                    for segment in action["segments"]:
                        segments.append(ActionSegment(segment["start_time"], segment["end_time"], segment["gripper_state"], segment["label"]))
                except Exception as e:
                    logging.error(f"Error analyzing segments: {e} for action: {action['description']}")

                action_obj = Action(
                    trajectory=trajectory,
                    preconditions=[p.replace("  ", "_").replace(".", "").replace(" ", "_") for p in pre_conditions],
                    postconditions=[p.replace("  ", "_").replace(".", "").replace(" ", "_") for p in post_conditions],
                    gripper_events=action.get('gripper_events', None),
                    label=action["description"].replace(" ", "_").replace(".", ""),
                    segments=segments,
                    folder_path=action_path
                )
                actions.append(action_obj)
                saved_folders.append(action_path)

            except Exception as e:
                logging.error(f"Error processing action {action['description']}: {e}")
                sys.exit(1)

        all_actions.append(actions)  # Collect all actions from all JSON files
    
    save_all_demonstrations_to_file(all_actions, "all_demonstrations.pkl")
    all_actions = load_all_demonstrations_from_file("all_demonstrations.pkl")
    print(f"Loaded {len(all_actions)} actions from all_demonstrations.pkl")
    print(f"First action: {all_actions[0]}")   
    action_list = []
    for demo in all_actions:
        demo_action_list = []
        for action in demo:
            added_str = ', '.join(str(c) for c in action.effect[0]) if action.effect[0] else '∅'
            removed_str = ', '.join(str(c) for c in action.effect[1]) if action.effect[1] else '∅'
            print(f"Pre-conditions: {action.preconditions}")
            print(f"Post-conditions: {action.postconditions}")
            print(f"Effect: (+{added_str} | -{removed_str})")
            print(f"Label: {action.label}\n")
            action_list.append([action.label, action.postconditions, action.preconditions])
    print(action_list)

    print(f"Fused {len(all_actions)} actions into {len(all_fused_actions)} actions across all demonstrations")
    
    base_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
    learn_and_save_dmps(all_actions, base_dir)

    print("Action list:")
    for action in action_list:
        print(action[0])  # action is already [label, postconditions, preconditions]
    
    print("Generating CSV file...")
    generate_csv([action_list], output_file=os.path.join(base_dir, "actions_preconditions.csv"))

    print("Building behavior tree...")
    root = build_behavior_tree_recursive(action_list)
    
    save_tree_to_file(root, filename=base_dir + "/behavior_tree.xml")
    print("Behavior tree saved successfully.")
