import json
import os
import argparse
from tqdm import tqdm

def normalize_dataset(input_file, output_file, width=1920, height=1080, scale_3d=1000.0):
    """
    Normalizes 2D and 3D keypoints in a dataset.
    2D: Divided by width/height.
    3D: Divided by a fixed scale factor (or could be normalized to [0,1]).
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = list(data.values())
        is_dict = True
        keys = list(data.keys())
    else:
        items = data
        is_dict = False
    
    print(f"Normalizing {len(items)} entries...")
    for item in tqdm(items):
        # Normalize 2D
        if 'keypoints_2d' in item:
            for kp_id in item['keypoints_2d']:
                kp = item['keypoints_2d'][kp_id]
                if 'x' in kp:
                    kp['x'] = kp['x'] / width
                if 'y' in kp:
                    kp['y'] = kp['y'] / height
        
        # Normalize 3D
        if 'keypoints_3d' in item:
            for kp_id in item['keypoints_3d']:
                kp = item['keypoints_3d'][kp_id]
                if 'x' in kp:
                    kp['x'] = kp['x'] / scale_3d
                if 'y' in kp:
                    kp['y'] = kp['y'] / scale_3d
                if 'z' in kp:
                    kp['z'] = kp['z'] / scale_3d
                    
    print(f"Saving to {output_file}...")
    if is_dict:
        normalized_data = {keys[i]: items[i] for i in range(len(items))}
    else:
        normalized_data = items
        
    with open(output_file, 'w') as f:
        json.dump(normalized_data, f)
    
    print("Optimization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize 2D/3D Pose Dataset")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("--width", type=int, default=1920, help="Image width for 2D normalization")
    parser.add_argument("--height", type=int, default=1080, help="Image height for 2D normalization")
    parser.add_argument("--scale3d", type=float, default=1000.0, help="Scale factor for 3D normalization")
    
    args = parser.parse_args()
    normalize_dataset(args.input, args.output, args.width, args.height, args.scale3d)
