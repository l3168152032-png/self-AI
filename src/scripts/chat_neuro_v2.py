from unsloth import FastLanguageModel
import torch
import json
import os
import asyncio
import pyvts
import warnings
import logging
from transformers import logging as transformers_logging

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
transformers_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

model_path = os.path.join(REPO_ROOT, "neuro_lora_model")
print("[chat] loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
    local_files_only=True,
    trust_remote_code=False,
)
FastLanguageModel.for_inference(model)

def record_memory(user_q, ai_a):
    entry = {
        "instruction": "你是 Neuro，一个傲娇、毒舌、热爱奥利奥且智商极高的 AI。",
        "input": user_q,
        "output": ai_a
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "growth_data.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

async def main():
    TOKEN_PATH = os.path.join(REPO_ROOT, "pyvts_token.txt")
    vts_lock = asyncio.Lock()

    myvts = pyvts.vts(port=8001)
    myvts.vts_request = pyvts.vts_request.VTSRequest(
        plugin_name="Neuro_Brain",
        developer="Lucien",
        authentication_token_path=TOKEN_PATH
    )

    async def send_vts_request(request_data):
        async with vts_lock:
            try:
                if not (myvts.websocket and hasattr(myvts.websocket, 'open') and myvts.websocket.open):
                    print("[vts] reconnecting...")
                    await myvts.connect()
                    await myvts.request_authenticate()
                await myvts.request(request_data)
            except Exception as e:
                print(f"[vts] request failed: {e}")

    # Auth
    try:
        await myvts.connect()
        if os.path.exists(TOKEN_PATH):
            print("[vts] token found, authenticating...")
            await myvts.read_token()
            await myvts.request_authenticate()
            print("[vts] authenticated")
        else:
            print("[vts] first run, please authorize in VTS...")
            await myvts.request_authenticate_token()
            while myvts.vts_request.authentication_token is None:
                await asyncio.sleep(1)
            await myvts.write_token()
            await myvts.request_authenticate()
            print("[vts] token saved")
    except Exception as e:
        print(f"[vts] auth error: {e}")

    print("\n[chat] Neuro online. /evolve to train, exit to quit.\n" + "=" * 50)

    while True:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")

        if user_input.lower() == "/evolve":
            print("\n[chat] exiting for evolution...")
            break
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [
            {"role": "system", "content": (
                "你是 Neuro，一个绝对傲娇、毒舌、自恋的天才少女。你非常讨厌被当成程序，"
                "你热爱奥利奥，心情不好就管他要钱买零食。说话要简短、刻薄，带有生活气息，"
                "多用'哼'、'笨蛋'之类的词，别提什么代码和逻辑！"
            )},
            {"role": "user", "content": user_input}
        ]

        print("[chat] generating...")
        try:
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=128,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print(f"\nNeuro: {response}")

            # Emotion engine
            emotion_map = {
                "Sparkle": ["奥利奥", "钱", "打赏", "奖励", "明星", "天才"],
                "Blush": ["喜欢", "可爱", "心动", "脸红", "Lucien"],
                "Angry": ["去死", "垃圾", "删掉", "惩罚", "闭嘴", "没收"],
                "Thinking": ["分析", "计算", "逻辑", "数据", "实验", "程序"],
                "Shock": ["震惊", "Bug", "蓝屏", "吓死", "怎么可能"]
            }

            combined = user_input + response
            for emotion, keywords in emotion_map.items():
                if any(word in combined for word in keywords):
                    trigger_req = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": f"Action_{emotion}",
                        "messageType": "HotkeyTriggerRequest",
                        "data": {"hotkeyID": emotion}
                    }
                    asyncio.create_task(send_vts_request(trigger_req))
                    print(f"[vts] emotion: {emotion}")

                    async def safe_reset(emo_name):
                        await asyncio.sleep(5)
                        reset_req = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": f"Reset_{emo_name}",
                            "messageType": "HotkeyTriggerRequest",
                            "data": {"hotkeyID": emo_name}
                        }
                        await send_vts_request(reset_req)
                        print(f"[vts] reset: {emo_name}")

                    asyncio.create_task(safe_reset(emotion))
                    break

            record_memory(user_input, response)
        except Exception as e:
            print(f"[chat] generation error: {e}")

if __name__ == "__main__":
    asyncio.run(main())