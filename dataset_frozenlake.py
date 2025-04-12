import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

PROMPT = """Task: Frozen Lake Shortest Path Planning

You are given an image of a grid-based environment. In this environment:
- An elf marks the starting position.
- A gift represents the goal.
- Some cells contain ice holes that are impassable for the elf.
- The elf can move in one of four directions only: "up", "down", "left", or "right". Each move transitions the elf by one cell in the corresponding absolute direction. Diagonal movement is not permitted.

Your task is to analyze the image and generate the shortest valid sequence of actions that moves the elf from the starting position to the goal without stepping into any ice holes.

Provide your final answer enclosed between <ANSWER> and </ANSWER>, for example: <ANSWER>right up up</ANSWER>. \n<image>"""


is_test = False

action_map = {
    0: "left",
    1: "down",
    2: "right",
    3: "up"
}

# Define the root directory where the images and JSON files are stored
root_dir = Path("frozenlake/optimal_with_distance")  # Relative path without leading slash
# Alternatively, use an absolute path that exists on your system
# root_dir = Path("C:/Users/Yi/Documents/GitHub/Qwen2.5/frozenlake/optimal_with_distance")
output_data = []

length_distributions = defaultdict(list)


# Loop through level directories (e.g., level0, level1, level2, level3)
for level_name in os.listdir(root_dir):
    level_path = root_dir / level_name
    
    assert level_path.is_dir()

    # Load the data.json file inside the level directory
    data_json_path = level_path / "data.json"

    assert data_json_path.exists()


    with open(data_json_path, 'r') as f:
        data = json.load(f)


    if is_test:
        subfolders = sorted(
            [p for p in level_path.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda x: int(x.name)
        )[1000:]
    else:
        subfolders = sorted(
            [p for p in level_path.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda x: int(x.name)
        )[:1000]

    # Loop through subdirectories (e.g., 0, 1, 2, ..., 1249)
    for subfolder in subfolders:
        # print(subfolder)

        assert subfolder.is_dir()

        folder_name = subfolder.name
        
        image_path = subfolder / "0.jpg"

        assert image_path.exists()

        for file in subfolder.iterdir():
            if file.is_file() and file.name != "0.jpg":
                file.unlink()

        # image_relative_path = f"{level_name}/{folder_name}/0.jpg"
        actions = [action_map[a] for a in data[folder_name]["actions"]]
        length_distributions[level_name].append(len(actions))

        output = "<ANSWER>" + " ".join(actions) + "</ANSWER>"

        conversation_data = {
            "image": str(image_path).replace("\\", "/"),
            "conversations": [
                {
                    "from": "human",
                    "value": PROMPT
                },
                {
                    "from": "gpt",
                    "value": output
                }
            ]
        }
        output_data.append(conversation_data)


plt.figure(figsize=(12, 6))
width = 0.2
all_lengths = sorted(set(length for lengths in length_distributions.values() for length in lengths))
x_indices = list(range(len(all_lengths)))

for i, (level_name, lengths) in enumerate(sorted(length_distributions.items())):
    counter = Counter(lengths)
    freqs = [counter.get(length, 0) for length in all_lengths]
    plt.bar([x + i * width for x in x_indices], freqs, width=width, label=level_name)

plt.xticks([x + width * (len(length_distributions) - 1) / 2 for x in x_indices], all_lengths)
plt.xlabel("Path Length")
plt.ylabel("Frequency")
plt.title("Optimal Path Length Distribution by Level (Bar Chart)")
plt.legend(title="Level")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# if is_test:
#     output_path = root_dir.parent / "test.json"
# else:
#     output_path = root_dir.parent / "train.json"
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(output_data, f, indent=2, ensure_ascii=False)

# print(f"Saved {len(output_data)} examples to {output_path}")

