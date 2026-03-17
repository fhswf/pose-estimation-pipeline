import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def merge_json_files(source_dirs, output_file):
    """
    Merges all JSON files from multiple source_dirs into a single list and saves to output_file.
    Deduplicates based on (source_video, frame_id).
    """
    merged_data_map = {} # Keyed by (video, frame) to handle duplicates
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
            continue

        # Get all .json files
        json_files = sorted(list(source_path.glob("*.json")))
        
        if not json_files:
            print(f"No JSON files found in {source_dir}.")
            continue

        print(f"Found {len(json_files)} files in {source_dir}. Merging...")

        # Using tqdm for progress tracking
        for fpath in tqdm(json_files, desc=f"Processing {source_path.name}"):
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    # Unique key for deduplication
                    key = (data.get("source_video"), data.get("frame_id"))
                    if key not in merged_data_map:
                        merged_data_map[key] = data
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    merged_data = list(merged_data_map.values())
    print(f"Saving merged data ({len(merged_data)} unique entries) to {output_file}...")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            json.dump(merged_data, f)
        print("Successfully saved merged file.")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate individual frame JSONs into a single dataset file.")
    parser.add_argument("--source", type=str, nargs='+', required=True, help="One or more directories containing individual JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output consolidated JSON file.")
    
    args = parser.parse_args()
    
    merge_json_files(args.source, args.output)
