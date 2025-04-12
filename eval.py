import json
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
# )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "output", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

with open("frozenlake/test.json", "r") as f:
    data = json.load(f)

results_path = Path("frozenlake/eval_results_low.json")

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
    prompt_text = example["conversations"][0]["value"]

    # 构造 messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 文本模板处理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 图像处理（你需要实现或替换 process_vision_info）
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 模型生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False  # 可根据需要改为 True
        )

    # 解码模型输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    results.append({
        "image": image_path,
        "prompt": prompt_text,
        "model_output": output_text
    })
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(image_path)
    print(output_text)
    print(example["conversations"][1]["value"])
    idx += 1