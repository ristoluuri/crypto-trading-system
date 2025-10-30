import os
import pandas as pd
import json
from multiprocessing import Pool, cpu_count

# === Base folders ===
# Folder containing original JSON files
json_base_folder = r"C:\Users\Nukes\Desktop\BTCUSDT2\src\data"
# Folder where converted Feather files will be stored
feather_base_folder = r"C:\Users\Nukes\Desktop\BTCUSDT2\src\data_feather"

# === Function to convert a single JSON file to Feather format ===
def json_to_feather(json_path):
    try:
        # Construct corresponding feather file path
        # Keeps folder structure relative to json_base_folder
        feather_path = os.path.join(
            feather_base_folder, os.path.relpath(json_path, json_base_folder)
        ).replace(".json", ".feather")

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(feather_path), exist_ok=True)

        # Read JSON file line by line (handles JSON Lines format)
        with open(json_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f if line.strip()]

        # Convert to Pandas DataFrame
        df = pd.DataFrame(data_list)
        # Save as Feather file
        df.to_feather(feather_path)
        return f"Converted: {json_path}"
    except Exception as e:
        # Return error message if conversion fails
        return f"Error converting {json_path}: {e}"

# === Main execution ===
if __name__ == "__main__":
    # Collect all JSON files in json_base_folder recursively
    all_json_files = []
    for root, dirs, files in os.walk(json_base_folder):
        for file in files:
            if file.endswith(".json"):
                all_json_files.append(os.path.join(root, file))

    # Use multiprocessing to convert files in parallel
    with Pool(cpu_count()) as pool:  # Use all available CPU cores
        # imap_unordered returns results as they complete
        for result in pool.imap_unordered(json_to_feather, all_json_files):
            print(result)  # Print conversion status or errors
