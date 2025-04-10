import os
import json
from pathlib import Path

# Define the root directory where the images and JSON files are stored
root_dir = Path("frozenlake/optimal_with_distance")  # Relative path without leading slash
# Alternatively, use an absolute path that exists on your system
# root_dir = Path("C:/Users/Yi/Documents/GitHub/Qwen2.5/frozenlake/optimal_with_distance")
output_data = []

# Loop through level directories (e.g., level0, level1, level2, level3)
for level_name in os.listdir(root_dir):
    level_path = root_dir / level_name
    
    assert level_path.is_dir()

    # Load the data.json file inside the level directory
    data_json_path = level_path / "data.json"

    assert data_json_path.exists()


    with open(data_json_path, 'r') as f:
        data = json.load(f)

    # Loop through subdirectories (e.g., 0, 1, 2, ..., 1249)
    for subfolder in sorted(
        [p for p in level_path.iterdir() if p.is_dir()],
        key=lambda x: int(x.name)
    ):
        assert subfolder.is_dir()

        folder_name = subfolder.name
        
        image_path = subfolder / "0.jpg"

        assert image_path.exists()

        # image_relative_path = f"{level_name}/{folder_name}/0.jpg"
        conversation_data = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "Your task is to analyze the image and generate the shortest valid sequence of actions that moves the character from its starting position to the goal, without stepping into any obstacle. \n<image>"
                },
                {
                    "from": "gpt",
                    "value": data[folder_name]["actions"]
                }
            ]
        }
        output_data.append(conversation_data)
        print(conversation_data)