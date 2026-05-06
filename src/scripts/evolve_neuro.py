import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
import gc
from unsloth import FastLanguageModel
import unsloth
unsloth.USE_FUSED_CE = False  # 禁用融合交叉熵，改用标准模式，虽然慢一点但稳

gc.collect()
torch.cuda.empty_cache()

# Resolve paths relative to repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

def _pick_growth_data_path():
    """优先读取 data/ 下的增长文件；兼容旧版根目录文件。"""
    candidates = [
        os.path.join(DATA_DIR, "growth_data.jsonl"),
        os.path.join(REPO_ROOT, "growth_data.jsonl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

# 1. 检查是否有新记忆
growth_data_path = _pick_growth_data_path()
if not os.path.exists(growth_data_path):
    print("[evolve] growth_data.jsonl not found, exit")
    exit()

# 2. 载入模型（开启训练模式）
print("[evolve] loading model...")
# 2. 载入模型（自动处理已有的 LoRA）

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = os.path.join(REPO_ROOT, "neuro_lora_model"),     max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = FastLanguageModel.for_training(model) 

print("[evolve] model loaded")

# 4. 载入并格式化数据
dataset = load_dataset("json", data_files=growth_data_path, split="train")

def format_prompt(sample):
    instruction = sample.get("instruction") or "你是 Neuro，一个直率、讨人喜欢、热爱奥利奥且智商极高的 AI。"
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    return {"text": text + tokenizer.eos_token}

dataset = dataset.map(format_prompt)

# 5. SFTTrainer
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
        
        fp16 = False,
        optim = "adamw_8bit",
        bf16 = True,  # RTX 4060 supports bf16
        logging_steps = 1,
        output_dir = os.path.join(REPO_ROOT, "evolution_outputs"),
        save_strategy = "no",
        gradient_checkpointing = True,
        max_grad_norm = 0.3,
    ),
)
print("[evolve] training...")
trainer.train()

# 6. Save LoRA
model.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model"))
tokenizer.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model"))

# 7. Archive growth data
history_out_path = os.path.join(DATA_DIR, "history_growth.jsonl")
base_history_path = os.path.join(REPO_ROOT, "history_growth.jsonl")
if not os.path.exists(history_out_path) and os.path.exists(base_history_path):
    history_out_path = base_history_path

if os.path.exists(growth_data_path):
    with open(growth_data_path, "r", encoding="utf-8") as f_src:
        new_data = f_src.read()

    if new_data.strip():
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(history_out_path, "a", encoding="utf-8") as f_dest:
            f_dest.write(new_data)
            f_dest.flush()
            os.fsync(f_dest.fileno())

    # rename-then-delete 防止中途崩溃丢数据
    bak_path = growth_data_path + ".bak"
    os.rename(growth_data_path, bak_path)
    os.remove(bak_path)
    print("[evolve] done")
else:
    print("[evolve] no growth_data.jsonl, skipped")