from unsloth import FastLanguageModel
import torch

# 1. 指向你训练好的成果文件夹
model_path = "neuro_persona_finished" 

# 2. 加载模型（原理：加载基础模型 + 合并你的性格补丁）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 切换到推理模式，速度更快

def ask_neuro(question):
    messages = [
        {"role": "system", "content": "你现在是 Neuro-sama。性格：极度聪明、毒舌、自恋。说话结尾带 Heart heart~"},
        {"role": "user", "content": question}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    # 生成回复
    outputs = model.generate(input_ids=inputs, max_new_tokens=128, temperature=0.8)
    response = tokenizer.batch_decode(outputs)[0]
    
    # 清洗掉系统提示词，只看回答
    return response.split("assistant\n")[-1].replace("<|im_end|>", "")

# 开启无限对话循环
print("--- Neuro-sama 已上线 (输入 'exit' 退出) ---")
while True:
    user_input = input("你: ")
    if user_input.lower() == 'exit': break
    print(f"Neuro: {ask_neuro(user_input)}")