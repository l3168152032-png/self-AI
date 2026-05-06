import os
os.environ["HF_HUB_OFFLINE"] = "0"

from unsloth import FastLanguageModel
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
lora_dir = os.path.join(REPO_ROOT, "neuro_lora_model")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_dir,
    load_in_4bit=True,
    local_files_only=True,
)

print("[merge] merging LoRA into 16-bit model...")
model.save_pretrained_merged(
    os.path.join(REPO_ROOT, "neuro_final_model"),
    tokenizer,
    save_method="merged_16bit",
)
print("[merge] done: neuro_final_model/")