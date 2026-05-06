import os
from unsloth import FastLanguageModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
checkpoint_path = os.path.join(REPO_ROOT, "outputs", "checkpoint-60")

print(f"[save] loading checkpoint: {checkpoint_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=2048,
    load_in_4bit=True,
)

print("[save] saving as neuro_lora_model...")
model.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model"))
tokenizer.save_pretrained(os.path.join(REPO_ROOT, "neuro_lora_model"))
print("[save] done")