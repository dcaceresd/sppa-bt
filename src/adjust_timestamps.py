import json
import cv2
import numpy as np
from PIL import Image
import copy
import os
import logging
import sys
from pathlib import Path

from sam_lang import GeometricRelationAnalyzer


def extract_frame(video_path, timestamp):
    """Extract frame from video at specific timestamp"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(round(timestamp * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

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

def has_human_in_frame(frame, vlm):
    """
    Check if a frame contains a human using VLM (Visual Language Model)
    """
    temp_path = "temp_frame.png"
    frame.save(temp_path)
    
    labels = "human. person. arm. hand."
    try:
        # Analyze image for humans
        image_pil = Image.open(temp_path).convert("RGB")
        results = vlm.model.predict([image_pil], [labels])
        # print(results)
        # Check if any human/person was detected
        if len(results[0]["scores"]) == 0:
            has_human = False
            print("No human detected")
        else:
            has_human = True if max(results[0]["scores"]) > 0.4 else False
            print(f"Has human: {has_human}, max score: {max(results[0]["scores"])}")
        
        return has_human
    except Exception as e:
        logging.error(f"Error analyzing frame for humans: {e}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def adjust_timestamps(json_path, video_path, max_attempts=50, time_step=0.2):
    """
    Adjust timestamps in JSON file to avoid frames with humans
    """
    vlm = GeometricRelationAnalyzer()
    
    with open(json_path) as f:
        config = json.load(f)
    
    if "demonstration" not in config or "actions" not in config["demonstration"]:
        print("Error: Invalid JSON structure")
        return
    
    modified = False
    
    def find_human_free_frame(frame_type, original_time, action):
        """Find a human-free frame for either start or end time"""
        original_seconds = time_to_seconds(original_time)
        current_seconds = original_seconds
        step_size = time_step
        
        for attempt in range(max_attempts):
            frame = extract_frame(video_path, current_seconds)
            if frame is None:
                print(f"Failed to extract {frame_type} frame at timestamp {original_time}")
                return None
                
            if not has_human_in_frame(frame, vlm):
                print(f"Found human-free {frame_type} frame at timestamp {original_time}")
                return current_seconds
                
            backward_seconds = original_seconds - step_size
            if backward_seconds >= 0:  # Ensure we don't go negative
                frame = extract_frame(video_path, backward_seconds)
                if frame is not None and not has_human_in_frame(frame, vlm):
                    print(f"Found human-free {frame_type} frame at timestamp {backward_seconds}")
                    return backward_seconds
                    
            forward_seconds = original_seconds + step_size
            frame = extract_frame(video_path, forward_seconds)
            if frame is not None and not has_human_in_frame(frame, vlm):
                print(f"Found human-free {frame_type} frame at timestamp {forward_seconds}")
                return forward_seconds
                
            step_size += time_step
            
        print(f"Warning: Could not find human-free {frame_type} frame for action at timestamp {original_time}")
        return None
    
    for action in config["demonstration"]["actions"]:
        if "start_time" not in action or "end_time" not in action:
            print(f"Skipping action without start_time or end_time")
            continue
            
        original_start_time = action["start_time"]
        original_end_time = action["end_time"]
        
        new_start_seconds = find_human_free_frame("start", original_start_time, action)
        new_end_seconds = find_human_free_frame("end", original_end_time, action)
        
        if new_start_seconds is not None and new_end_seconds is not None:
            new_start_time = f"{int(new_start_seconds // 60):02d}:{new_start_seconds % 60:.2f}"
            new_end_time = f"{int(new_end_seconds // 60):02d}:{new_end_seconds % 60:.2f}"
            
            action["start_time"] = new_start_time
            action["end_time"] = new_end_time
            print(f"Adjusted times from {original_start_time} to {new_start_time} and {original_end_time} to {new_end_time}")
            modified = True
    
    if modified:
        output_path = json_path.replace('.json', '_adjusted.json')
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved modified JSON to {output_path}")
    else:
        print("No timestamps needed adjustment")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python adjust_timestamps.py <json_path>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    with open(json_path) as f:
        config = json.load(f)
    
    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_path = str(Path(json_path).with_suffix(''))
    video_path = f"{base_path}.mp4"
    print(f"Processing video from: {video_path}")


    adjust_timestamps(json_path, video_path)
