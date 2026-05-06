from unsloth import FastLanguageModel
import torch
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
model_path = os.path.join(REPO_ROOT, "neuro_persona_finished")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def ask_neuro(question):
    messages = [
        {"role": "system", "content": "你现在是 Neuro-sama。性格：极度聪明、毒舌、自恋。说话结尾带 Heart heart~"},
        {"role": "user", "content": question}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=128, temperature=0.8)
    response = tokenizer.batch_decode(outputs)[0]
    return response.split("assistant\n")[-1].replace("<|im_end|>", "")

print("--- Neuro-sama online (type 'exit' to quit) ---")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    print(f"Neuro: {ask_neuro(user_input)}")