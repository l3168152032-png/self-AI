import os
import torch
import unsloth.models._utils
from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# Resolve paths relative to repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# --- 1. Env ---
_triton_path = os.environ.get("TRITON_PTXAS_PATH", "")
if not _triton_path:
    _cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    _ptxas_candidates = [
        os.path.join(_cuda_home, "bin", "ptxas.exe"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\ptxas.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\ptxas.exe",
    ]
    for _c in _ptxas_candidates:
        if os.path.exists(_c):
            os.environ["TRITON_PTXAS_PATH"] = _c
            break
unsloth.models._utils.get_statistics = lambda *args, **kwargs: None
os.environ["UNSLOTH_USE_TRITON"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

# --- 2. Load model ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/qwen2.5-7b-instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)

# --- 3. LoRA config ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth",
    bias = "none",
)

# --- 4. Dataset ---
def formatting_prompts_func(examples):
    instructions = examples.get("instruction") or []
    inputs       = examples.get("input") or []
    outputs      = examples.get("output") or []
    texts = []
    max_len = max(len(instructions), len(inputs), len(outputs))
    default_inst = "你是 Neuro，一个直率、讨人喜欢、热爱奥利奥且智商极高的 AI。"
    for i in range(max_len):
        inst = instructions[i] if i < len(instructions) and instructions[i] else default_inst
        inp = inputs[i] if i < len(inputs) else ""
        out = outputs[i] if i < len(outputs) else None
        if out is not None:
            texts.append(f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}" + tokenizer.eos_token)
    if not texts:
        return ["### Instruction:\nSystem\n\n### Response:\nData Loading Failed" + tokenizer.eos_token]
    return texts

data_sources = []

data_sources.append(load_dataset("json", data_files=os.path.join(DATA_DIR, "neuro_train.jsonl"), split="train"))

history_path = os.path.join(DATA_DIR, "history_growth.jsonl")
if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
    data_sources.append(load_dataset("json", data_files=history_path, split="train"))
    print("[train] loaded history_growth.jsonl")

growth_path = os.path.join(DATA_DIR, "growth_data.jsonl")
if os.path.exists(growth_path) and os.path.getsize(growth_path) > 0:
    data_sources.append(load_dataset("json", data_files=growth_path, split="train"))
    print("[train] loaded growth_data.jsonl")

combined_dataset = concatenate_datasets(data_sources)
combined_dataset = combined_dataset.shuffle(seed=3407)
print(f"[train] samples: {len(combined_dataset)}")

# --- 5. Trainer ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    formatting_func = formatting_prompts_func,
    max_seq_length = 2048,
    args = TrainingArguments(
        gradient_checkpointing = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 250, 
        learning_rate = 3e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 1,
        log_level = "info",
        disable_tqdm = False,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        report_to = "none",
    ),
)

# --- 6. Run ---
print("[train] training...")
trainer.train()

model.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model")) 
tokenizer.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model"))
print("[train] done")
