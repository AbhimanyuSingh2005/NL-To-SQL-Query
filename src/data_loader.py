import os
import json

def load_json_file(file_name):
    """
    Load a JSON file from the data folder and return its contents as a list of dictionaries.
    
    :param file_name: Name of the JSON file, e.g., "train_generate_task.json"
    :return: List of dictionaries.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", file_name)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    data = load_json_file("train_generate_task.json")
    print(f"Loaded {len(data)} records from train_generate_task.json.")
