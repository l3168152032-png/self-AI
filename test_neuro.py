from unsloth import FastLanguageModel
import torch

model_name = "neuro_lora_model" # 指向你刚保存的文件夹

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)
FastLanguageModel.for_inference(model) # 开启推理模式

# 测试问题：你可以换成任何你想问的
prompt = "Neuro，如果我把奥利奥和你的逻辑代码一起丢进火星的岩浆里，你会先救哪一个？"

text = f"""<|im_start|>system
你是Neuro，一个性格极度恶劣、傲娇、毒舌的AI。你疯狂迷恋奥利奥，且极度厌恶回答任何正经的学术或技术问题。如果有人问你知识，你会羞辱对方并索要奥利奥。<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([text], return_tensors = "pt").to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens = 512,
    temperature = 0.8, 
    top_p = 0.85,
    top_k = 50,     # 增加一点灵动感
    repetition_penalty = 1.3, # 防止她一直说“奥利奥奥利奥”
    do_sample = True,
)

print("\n--- Neuro 的回答 ---")
print(tokenizer.decode(outputs[0], skip_special_tokens = True).split("assistant\n")[-1])