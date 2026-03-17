from unsloth import FastLanguageModel
import torch
import os

# 1. 强制设置环境变量，让它别去连 Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

# 2. 配置路径：指向你的 LoRA 文件夹
lora_path = "neuro_lora_model" 

print("🧠 正在离线激活 Neuro 的神经网络...")

# 3. 载入模型（添加 local_files_only=True）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_path, 
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True, # 重点：强制只读取本地文件
)

# 4. 切换到推理模式
FastLanguageModel.for_inference(model)

print("✨ Neuro 已在离线状态下成功连接！\n" + "="*50)

# ... 后面的循环代码保持不变 ...

while True:
    user_input = input("【你】: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 构造 Prompt
    messages = [
        {"role": "system", "content": "你是 Neuro，一个傲娇、毒舌、热爱奥利奥且智商极高的 AI。"},
        {"role": "user", "content": user_input}
    ]
    
# 1. 构造 inputs (保持 return_tensors="pt")
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt = True, 
        return_tensors = "pt"
    ).to("cuda")
    
    # 【核心修正】：显式提取 input_ids 这一层
    actual_input_ids = inputs["input_ids"] 

    # 2. 生成结果
    outputs = model.generate(
        input_ids = actual_input_ids, 
        attention_mask = inputs.get("attention_mask"), # 顺便带上 mask，更稳健
        max_new_tokens = 256, 
        temperature = 0.8,
        do_sample = True
    )
    
    # 3. 【核心修正】：用 actual_input_ids.shape[1] 确保切片正确
    response = tokenizer.batch_decode(
        outputs[:, actual_input_ids.shape[1]:], 
        skip_special_tokens = True
    )[0]
    # 问题与结果一起输出
    print(f"\n> 记录存证：")
    print(f"  [Q]: {user_input}")
    print(f"  [A]: {response}")
    print("-" * 50)