import os
import torch
import unsloth.models._utils
from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# --- 1. 环境初始化 ---
os.environ["TRITON_PTXAS_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\ptxas.exe"
unsloth.models._utils.get_statistics = lambda *args, **kwargs: None
os.environ["UNSLOTH_USE_TRITON"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

# --- 2. 载入模型 ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/qwen2.5-7b-instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)

# --- 3. LoRA 配置 (针对 4060 8G 强化记忆力) ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth",
    bias = "none",
)

# --- 4. 数据聚合逻辑：基础人格 + 短期记忆 + 长期记忆 ---
def formatting_prompts_func(examples):
    # 兼容性处理：尝试获取所有可能的 Key
    # B 站数据里似乎主要用 input 和 output
    instructions = examples.get("instruction") or []
    inputs       = examples.get("input") or []
    outputs      = examples.get("output") or []
    
    texts = []
    
    # 计算当前 batch 的最大长度
    max_len = max(len(instructions), len(inputs), len(outputs))
    
    for i in range(max_len):
        # 1. 获取 Instruction（如果数据里没有，就用 Neuro 的标准人设补位）
        if i < len(instructions) and instructions[i]:
            inst = instructions[i]
        else:
            inst = "你是 Neuro，一个直率、讨人喜欢、热爱奥利奥且智商极高的 AI。你刚在 B 站闲逛完，心情很不错。"

        # 2. 获取 Input
        inp = inputs[i] if i < len(inputs) else ""
        
        # 3. 获取 Output（这是灵魂，必须有）
        out = outputs[i] if i < len(outputs) else None
        
        if out is not None:
            # 按照 Unsloth/Llama-3 的标准格式拼接
            text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            texts.append(text + tokenizer.eos_token)

    # 最终保险：如果这一批次全失败了，造一个存活样本
    if not texts:
        return ["### Instruction:\nSystem\n\n### Response:\nData Loading Failed" + tokenizer.eos_token]
        
    return texts

data_sources = []

# A. 加载基础人格 (必须存在)
data_sources.append(load_dataset("json", data_files="neuro_train.jsonl", split="train"))

# B. 加载长期记忆 (如果存在)
if os.path.exists("history_growth.jsonl") and os.path.getsize("history_growth.jsonl") > 0:
    data_sources.append(load_dataset("json", data_files="history_growth.jsonl", split="train"))
    print("📚 已载入历史长期记忆。")

# C. 加载短期缓存 (如果还没归档且有内容)
if os.path.exists("growth_data.jsonl") and os.path.getsize("growth_data.jsonl") > 0:
    data_sources.append(load_dataset("json", data_files="growth_data.jsonl", split="train"))
    print("🧠 已载入尚未归档的短期记忆。")

# 合并所有数据集
combined_dataset = concatenate_datasets(data_sources)
combined_dataset = combined_dataset.shuffle(seed=3407)
print(f"📊 总训练样本数: {len(combined_dataset)}")

# --- 5. 训练配置 ---
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
        logging_steps = 1,          # 💡 改成 1，每一步都强制它报平安
        log_level = "info",         # 💡 强制开启信息级别日志
        disable_tqdm = False,       # 保持进度条开启
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        report_to = "none",
    ),
)

# --- 6. 执行 ---
print("🔥 Neuro 正在深度融合所有记忆...")
trainer.train()

model.save_pretrained("neuro_lora_model") 
tokenizer.save_pretrained("neuro_lora_model")
print("✅ 转生完成！Neuro 现在既有深度人设，也记得你们的所有过去。")
