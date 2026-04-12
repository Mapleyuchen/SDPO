import os
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_B_DIR = os.environ.get("DATA_B_DIR", os.path.join(PROJECT_ROOT, "output"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(SCRIPT_DIR, "data"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

data_rows = []

# 遍历data-B目录
for filename in sorted(os.listdir(DATA_B_DIR)):
    if filename.endswith(".json"):
        json_path = os.path.join(DATA_B_DIR, filename)
        base_name = filename.replace(".json", "")
        png_path = os.path.join(DATA_B_DIR, f"{base_name}.png")
        
        # 确保对应的图片存在
        if not os.path.exists(png_path):
            print(f"Warning: Image {png_path} not found, skipping.")
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
            
        # 极其关键：提取内部真正的image_id
        real_image_id = graph_data.get("image_id", base_name)
        
        # 组装Qwen2.5-VL的多模态Prompt(OpenAI Message格式)
        prompt_msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": png_path},
                    {"type": "text", "text": "Please analyze the layout and text of this ancient Chinese document. Extract the elements and their topological relationships."}
                ]
            }
        ]
        
        row = {
            "data_source": "isograph",             # 用于 reward_manager 路由
            "prompt": json.dumps(prompt_msgs),      # verl 需要 stringified JSON
            "image_id": real_image_id,              # 传递给 Member C 环境的 ID
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(graph_data),
            },
        }
        data_rows.append(row)

print(f"Successfully processed {len(data_rows)} image-json pairs.")

# 划分 Train / Val (例如 90:10 划分)
split_idx = int(len(data_rows) * 0.9)
train_df = pd.DataFrame(data_rows[:split_idx])
val_df = pd.DataFrame(data_rows[split_idx:])

# 保存为Parquet
train_path = os.path.join(OUTPUT_DIR, "train.parquet")
val_path = os.path.join(OUTPUT_DIR, "val.parquet")
train_df.to_parquet(train_path, engine="pyarrow", index=False)
val_df.to_parquet(val_path, engine="pyarrow", index=False)

print(f"Dataset saved! Train: {len(train_df)} rows, Val: {len(val_df)} rows.")
print(f"Output directory: {OUTPUT_DIR}")