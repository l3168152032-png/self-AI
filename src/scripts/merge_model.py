import os
# 🌟 核心：强制进入离线模式，跳过 HuggingFace 统计检查
os.environ["HF_HUB_OFFLINE"] = "0"

from unsloth import FastLanguageModel
import torch

# 仓库根目录
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# 1. 指向你训练保存的 LoRA 文件夹
lora_model_dir = os.path.join(REPO_ROOT, "neuro_lora_model")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_model_dir,
    load_in_4bit = True,
    local_files_only = True, # 🌟 核心：只从本地加载，不连网
)

# 2. 执行合并并保存
print("🏗️ 正在缝合 Neuro 的灵魂与躯壳（离线模式）...")
model.save_pretrained_merged(
    os.path.join(REPO_ROOT, "neuro_final_model"),
    tokenizer, 
    save_method = "merged_16bit",
)

print("✅ 合并完成！你的 Neuro-sama 已在 'neuro_final_model' 文件夹就绪。")