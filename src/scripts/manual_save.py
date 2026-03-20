import os
from unsloth import FastLanguageModel

# 1. 找到刚才训练产生的最近的一个检查点文件夹
# 通常在你的 output_dir（"outputs"）文件夹里，名字叫 checkpoint-40 之类的
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
checkpoint_path = os.path.join(REPO_ROOT, "outputs", "checkpoint-60") # 请去你的 outputs 文件夹看一眼具体的数字并修改这里

print(f"📦 正在从 {checkpoint_path} 加载灵魂碎片...")

# 2. 加载这个检查点的 LoRA 权重
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint_path, # 重点：这里填检查点的路径
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. 正式保存为你最终的名字
print("💾 正在整合并保存为最终版 neuro_lora_model...")
model.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model"))
tokenizer.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model"))

print("✨ 搞定！现在你可以运行 test_neuro.py 了。")