from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import sys
import json
import os
from pathlib import Path
import os
import argparse
from typing import Dict, Any, List

os.environ['MODELSCOPE_CACHE'] = '/sppa-bt/qwen_cache' 

MODEL_CACHE = "qwen_cache"
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# Initialize model and processor globally to avoid reloading for each video
model = None
processor = None

def initialize_models():
    global model, processor
    if model is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        print(f"Model loaded on device: {model.device}")
        model.gradient_checkpointing_enable()
    
    if processor is None:
        min_pixels = 128*28*28
        max_pixels = 256*28*28
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels, 
            cache_dir=MODEL_CACHE, 
            use_fast=True
        )
    
    return model, processor

def load_prompts() -> dict:
    """Load prompts from the JSON file in the config directory."""
    prompts_path = Path(__file__).parent.parent / 'config' / 'prompts.json'
    with open(prompts_path, 'r') as f:
        return json.load(f)

def get_conditioned_prompt(previous_output: str = '') -> str:
    """Generate a conditioned prompt based on previous output."""
    prompts = load_prompts()
    return prompts['conditioned_base'].format(previous_output=previous_output)

def get_prompts(previous_output: str = '') -> dict:
    """Get all prompt variations"""
    prompts = load_prompts()
    return {
        'basic': prompts['basic'],
        'detailed': prompts['detailed'],
        'objects_prompt': prompts['objects_prompt'],
        'conditioned': get_conditioned_prompt(previous_output)
    }

def process_video(video_path: str, fps_set: int = 2) -> Dict[str, Any]:
    """Process a single video file and return the analysis results."""
    print(f"\nProcessing video: {video_path}")
    video_uri = "file://" + os.path.abspath(video_path)
    prompts = get_prompts()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_uri, "fps": fps_set},
                {"type": "text", "text": prompts["detailed"]}
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    
    second_per_grid_ts = inputs.pop('second_per_grid_ts')
    second_per_grid_ts = [float(s) for s in second_per_grid_ts]
    inputs.update({'second_per_grid_ts': second_per_grid_ts})
    inputs = inputs.to(model.device)
    
    # Run inference
    video_tokens = inputs.input_ids.numel()
    print(f"Number of video tokens: {video_tokens}")
    if video_tokens > 24000:
        print("WARNING: Number of video tokens exceeds 24k! This might affect performance or cause issues.")
    
    max_tokens = 100000
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    try:
        json_string = output_text[0].strip("'").strip('\n').strip('`').strip().strip('json')
        actions = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding actions JSON: {e}")
        actions = {"error": f"Failed to parse actions: {str(e)}"}
    
    # Process objects detection
    messages_2 = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_uri, "fps": fps_set},
                {"type": "text", "text": prompts["objects_prompt"]}
            ],
        }
    ]
    
    text_2 = processor.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    image_inputs_2, video_inputs_2, video_kwargs_2 = process_vision_info(messages_2, return_video_kwargs=True)
    
    inputs_2 = processor(
        text=[text_2],
        images=image_inputs_2,
        videos=video_inputs_2,
        padding=True,
        return_tensors="pt",
        **video_kwargs_2,
    )
    
    second_per_grid_ts = inputs_2.pop('second_per_grid_ts')
    second_per_grid_ts = [float(s) for s in second_per_grid_ts]
    inputs_2.update({'second_per_grid_ts': second_per_grid_ts})
    inputs_2 = inputs_2.to(model.device)
    
    # Run inference for objects
    generated_ids_2 = model.generate(**inputs_2, max_new_tokens=100000)
    generated_ids_trimmed_2 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_2.input_ids, generated_ids_2)]
    output_text_2 = processor.batch_decode(generated_ids_trimmed_2, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    try:
        json_string_2 = output_text_2[0].strip("'").strip('\n').strip('`').strip().strip('json')
        detected_objects = json.loads(json_string_2)
    except json.JSONDecodeError as e:
        print(f"Error decoding objects JSON: {e}")
        detected_objects = {"error": f"Failed to parse objects: {str(e)}"}
    
    # Prepare output data
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_data = {
        "vlm": {
            "model_info": {
                "model_name": MODEL_ID,
                "device": str(model.device),
                "dtype": str(model.dtype)
            },
            "input_parameters": {
                "prompt": prompts["detailed"],
                "video_path": video_path,
                "fps": video_kwargs.get('fps', None),
                "frame_stride": video_kwargs.get('frame_stride', None)
            },
            "generation_parameters": {
                "max_new_tokens": max_tokens,
                "temperature": getattr(model.generation_config, 'temperature', None),
                "top_p": getattr(model.generation_config, 'top_p', None)
            },
        },
        "demonstration": {
            "video_path": video_path,
            "gripper": f"{base_name}_gripper_status.csv",
            "trajectories": f"{base_name}_ee_trajectory.csv",
            "actions": actions,
            "detected_objects": detected_objects,
        }
    }
    
    return output_data

def save_results(output_data: Dict[str, Any], video_path: str) -> str:
    """Save the analysis results to a JSON file."""
    input_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}.json"
    output_path = os.path.join(input_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=lambda x: str(x))
    
    print(f"Saved analysis results to {output_path}")
    return output_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process video(s) for action segmentation.')
    parser.add_argument('videos', nargs='+', help='One or more video files to process')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second for processing (default: 1)')
    args = parser.parse_args()
    
    # Initialize models
    model, processor = initialize_models()
    
    # Process each video
    for video_path in args.videos:
        if not os.path.isfile(video_path):
            print(f"Error: Video file not found: {video_path}")
            continue
            
        try:
            # Process the video
            output_data = process_video(video_path, args.fps)
            
            # Save the results
            output_path = save_results(output_data, video_path)
            print(f"Successfully processed {video_path} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()