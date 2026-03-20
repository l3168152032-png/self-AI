from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
import torch
import gc
import unsloth
unsloth.USE_FUSED_CE = False  # 禁用融合交叉熵，改用标准模式，虽然慢一点但稳
# 强力清理残留显存
gc.collect()
torch.cuda.empty_cache()

# 告诉 Unsloth 更加节俭地使用显存
from unsloth import FastLanguageModel

# 目录基于脚本位置解析，避免工作目录变化导致找不到记忆文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def _pick_growth_data_path():
    """优先读取 data/ 下的增长文件；兼容旧版根目录文件。"""
    candidates = [
        os.path.join(DATA_DIR, "growth_data.jsonl"),
        os.path.join(BASE_DIR, "growth_data.jsonl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

# 1. 检查是否有新记忆
growth_data_path = _pick_growth_data_path()
if not os.path.exists(growth_data_path):
    print("❌ 找不到 growth_data.jsonl，Neuro 觉得没什么好学的。")
    exit()

# 2. 载入模型（开启训练模式）
print("🧬 Neuro 进入深度思考（进化）状态...")
# 2. 载入模型（自动处理已有的 LoRA）
print("🧬 Neuro 正在读取现有的神经网络神经元...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "neuro_lora_model", # 你的模型文件夹
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)

if tokenizer.pad_token is None:
    # 强制获取 eos_token 的字符串内容
    tokenizer.pad_token = tokenizer.eos_token
    # 如果还是报错，直接暴力指定字符串：
    # tokenizer.pad_token = "<|endoftext|>"
# 3. 关键修改：不再调用 get_peft_model，因为 LoRA 已经加载好了
# 我们直接告诉模型开启训练模式即可
model = FastLanguageModel.for_training(model) 

print("✅ 已接管现有适配器，准备注入新记忆...")

# 4. 载入并格式化数据
dataset = load_dataset("json", data_files=growth_data_path, split="train")

def format_prompt(sample):
    # 尝试获取 instruction，如果没有，就用默认的理性人设
    instruction = sample.get("instruction")
    if not instruction:
        instruction = "你是 Neuro，一个直率、讨人喜欢、热爱奥利奥且智商极高的 AI。你刚在 B 站闲逛完，心情很不错。"
    
    # 获取 input 和 output
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")
    
    # 拼接模板
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    
    # 别忘了加上结束符，否则 Neuro 会变成话痨停不下来
    return { "text" : text + tokenizer.eos_token }

dataset = dataset.map(format_prompt)

# 5. 极速微调参数
# --- 修正后的 SFTTrainer 调用 ---
# --- 修正后的精度参数 ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_steps = 30,
        learning_rate = 1e-5,
        
        # --- 修改这里 ---
        fp16 = False,   
        optim = "adamw_8bit",        # 将 fp16 设为 False
        bf16 = True,            # 将 bf16 设为 True (4060 支持这个)
        # ----------------
        logging_steps = 1,
        output_dir = "evolution_outputs",
        save_strategy = "no",
        gradient_checkpointing = True,   # 开启梯度检查点，用计算换空间
        max_grad_norm = 0.3,
    ),
)
# ------------------------------
print("✨ 正在将记忆刻入神经网络...")
trainer.train()

# 6. 覆盖保存
model.save_pretrained("neuro_lora_model")
tokenizer.save_pretrained("neuro_lora_model")

# 7. 清理：将旧记忆归档，防止重复学习
# --- 修正后的归档逻辑 ---
import os

history_out_path = os.path.join(DATA_DIR, "history_growth.jsonl")
base_history_path = os.path.join(BASE_DIR, "history_growth.jsonl")
if not os.path.exists(history_out_path) and os.path.exists(base_history_path):
    history_out_path = base_history_path

if os.path.exists(growth_data_path):
    # 如果历史文件已存在，就把新内容接在后面
    with open(growth_data_path, "r", encoding="utf-8") as f_src:
        new_data = f_src.read()
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(history_out_path, "a", encoding="utf-8") as f_dest:
        f_dest.write(new_data)
    
    # 删除已经处理完的临时增长文件
    os.remove(growth_data_path)
    print("✅ 进化完成！新记忆已成功合并至 history_growth.jsonl。")
else:
    print("⚠️ 未发现 growth_data.jsonl，跳过归档。")