import base64
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from google import genai

PROMPT = """Task: Frozen Lake Shortest Path Planning

You are given an image of a grid-based environment. In this environment:
- An elf marks the starting position.
- A gift represents the goal.
- Some cells contain ice holes that are impassable for the elf.
- The elf can move in one of four directions only: "up", "down", "left", or "right". Each move transitions the elf by one cell in the corresponding absolute direction. Diagonal movement is not permitted.

Your task is to analyze the image and generate the shortest valid sequence of actions that moves the elf from the starting position to the goal without stepping into any ice holes.

Do not include any explanation. Only provide your final answer enclosed between <ANSWER> and </ANSWER>, for example: <ANSWER>right up up</ANSWER>."""


PROMPT_COT = """Task: Frozen Lake Shortest Path Planning

You are given an image of a grid-based environment. In this environment:
- An elf marks the starting position.
- A gift represents the goal.
- Some cells contain ice holes that are impassable for the elf.
- The elf can move in one of four directions only: "up", "down", "left", or "right". Each move transitions the elf by one cell in the corresponding absolute direction. Diagonal movement is not permitted.

Your task is to analyze the image and generate the shortest valid sequence of actions that moves the elf from the starting position to the goal without stepping into any ice holes.

Let's think step by step. Provide your final answer enclosed between <ANSWER> and </ANSWER>, for example: <ANSWER>right up up</ANSWER>."""

client = genai.Client()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


with open("frozenlake/test.json", "r") as f:
    data = json.load(f)

results_path = Path("frozenlake/eval_results_gemini_direct.json")

if results_path.exists():
    with open(results_path, "r") as f:
        results = json.load(f)
else:
    results = []

evaluated_images = set(r["image"] for r in results)

idx = 1
for example in tqdm(data):
    image_path = example["image"]
    if image_path in evaluated_images:
        print(f"Skipping already evaluated image: {image_path}")
        continue

    print(f"Evaluating data {idx}.")
    prompt_text = PROMPT

    image = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt_text, image]
    )

    output_text = response.text

    print(output_text)

    results.append({
        "image": image_path,
        "prompt": prompt_text,
        "model_output": output_text
    })
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(image_path)
    print(output_text)
    
    idx += 1
    if idx == 50:
        break

    

