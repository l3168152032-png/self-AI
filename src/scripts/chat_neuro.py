from unsloth import FastLanguageModel
import torch
import os

os.environ["HF_HUB_OFFLINE"] = "1"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
lora_path = os.path.join(REPO_ROOT, "neuro_lora_model")

print("[chat] loading model (offline)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_path,
    max_seq_length=2048,
    load_in_4bit=True,
    local_files_only=True,
)
FastLanguageModel.for_inference(model)
print("[chat] ready\n" + "=" * 50)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [
        {"role": "system", "content": "你是 Neuro，一个傲娇、毒舌、热爱奥利奥且智商极高的 AI。"},
        {"role": "user", "content": user_input}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    actual_input_ids = inputs["input_ids"]
    outputs = model.generate(
        input_ids=actual_input_ids,
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=256,
        temperature=0.8,
        do_sample=True
    )

    response = tokenizer.batch_decode(
        outputs[:, actual_input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    print(f"\n[Q]: {user_input}")
    print(f"[A]: {response}")
    print("-" * 50)