import os
import json
import argparse
from pathlib import Path

def get_coordinate_from_state(state, level):
    assert state < level*level, "state must be less than level*level"
    row = state // level
    col = state % level
    return (row, col)

def get_state_from_coordinate(coordinate, level):
    return coordinate[0] * level + coordinate[1]

def apply_action_sequence(start_coord, actions, level, target_pos, distance_dict):
    """Apply action sequence and return path with validity of each move."""

    move_delta = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    actions = [a.strip().lower() for a in actions]

    path = []
    current_coord = start_coord
    for action in actions:
        if action not in move_delta:
            path.append({"coord": current_coord, "valid": False})
            continue

        dr, dc = move_delta[action]
        next_coord = (current_coord[0] + dr, current_coord[1] + dc)

        # Check if the move is within bounds
        if 0 <= next_coord[0] < level and 0 <= next_coord[1] < level:
            next_state = str(get_state_from_coordinate(next_coord, level))

            if next_state in distance_dict or next_state == target_pos:
                valid = True
                current_coord = next_coord
            else:
                valid = False
        else:
            valid = False

        path.append({"coord": next_coord, "valid": valid})

    return path

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT model output on FrozenLake task.")
    parser.add_argument("--prefix", type=str, default="frozenlake", help="Directory prefix where the eval file is stored (default: frozenlake)")
    parser.add_argument("--name", type=str, required=True, help="Filename of the evaluation result (with or without .json)")
    args = parser.parse_args()

    # Ensure .json extension is added if not present
    eval_filename = args.name if args.name.endswith(".json") else args.name + ".json"
    eval_file = Path(args.prefix) / eval_filename

    root_dir = Path("frozenlake/optimal_with_distance")

    # Load model predictions
    with open(eval_file, "r") as f:
        eval_results = json.load(f)

    correct = 0
    total = 0
    level_accuracy = {}

    level_to_offset = {
        "level3": 0,
        "level4": 250,
        "level5": 500,
        "level6": 750,
    }

    for level_name in sorted(os.listdir(root_dir)):
        level = int(level_name[-1])  # level3 -> 3, etc.
        level_path = root_dir / level_name
        if not level_path.is_dir():
            continue

        print(f"Evaluating {level_name}...")

        data_path = level_path / "data_distance_map.json"
        with open(data_path, "r") as f:
            data_map = json.load(f)

        offset = level_to_offset[level_name]
        level_correct = 0
        level_total = 0

        for i in range(1000, 1250):
            subfolder = str(i)
            assert subfolder in data_map

            distance_dict = data_map[subfolder]["distance_map"]
            start_pos = str(data_map[subfolder]["start_pos"])
            target_pos = str(data_map[subfolder]["target_pos"])

            start_coordinate = get_coordinate_from_state(int(start_pos), level)
            assert start_pos in distance_dict
            true_distance = distance_dict[start_pos]

            prediction_entry = eval_results[offset + (i - 1000)]
            pred_text = prediction_entry["model_output"]

            if "<ANSWER>" in pred_text and "</ANSWER>" in pred_text:
                answer = pred_text.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
                pred_actions = answer.split()
            else:
                pred_actions = []

            path = apply_action_sequence(start_coordinate, pred_actions, level, target_pos, distance_dict)
            all_valid = all(step["valid"] for step in path)

            if all_valid:
                path_length_matches = len(path) == true_distance
                final_coord = path[-1]["coord"]
                final_state = get_state_from_coordinate(final_coord, level)
                target_state = int(data_map[subfolder]["target_pos"])
                ends_correctly = final_state == target_state

                if path_length_matches and ends_correctly:
                    print(f"Correct: {level_name} {subfolder} -> {pred_actions} (length: {len(path)}, true_distance: {true_distance}), correct {level_correct+1}")
                    correct += 1
                    level_correct += 1
            total += 1
            level_total += 1

        level_acc = level_correct / level_total if level_total > 0 else 0
        level_accuracy[level_name] = (level_acc, level_correct, level_total)
        print(f"{level_name} Accuracy: {level_acc:.4f} ({level_correct}/{level_total})")

    avg_acc = sum(acc for acc, _, _ in level_accuracy.values()) / len(level_accuracy)
    print("\n==== Final Summary ====")
    for level_name, (acc, corr, tot) in level_accuracy.items():
        print(f"{level_name:<7}: {acc*100:.4f} ({corr}/{tot})")
    print(f"{'Average':<7}: {avg_acc*100:.4f}")

if __name__ == "__main__":
    main()