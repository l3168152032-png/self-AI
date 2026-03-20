from unsloth import FastLanguageModel
import torch
import json
import os
import asyncio
import pyvts
import warnings
import logging
from transformers import logging as transformers_logging

# --- 0. 基础环境配置 ---
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
transformers_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

# 目录基于脚本位置解析，避免从其它工作目录运行导致找不到记忆文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 1. 模型加载 (指向你的 LoRA 文件夹)
model_path = "neuro_lora_model" 

print("🧠 正在唤醒 Neuro 的长期记忆 (Unsloth 加速版)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
    trust_remote_code = False,
)
FastLanguageModel.for_inference(model)

# 2. 记忆存储函数
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

# 3. 异步主逻辑
async def main():
    # --- 🔐 初始化 VTS 连接 ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TOKEN_PATH = os.path.join(BASE_DIR, "pyvts_token.txt")
    vts_lock = asyncio.Lock()
    
    myvts = pyvts.vts(port=8001)
    myvts.vts_request = pyvts.vts_request.VTSRequest(
        plugin_name="Neuro_Brain",
        developer="Lucien",
        authentication_token_path=TOKEN_PATH
    )

    # --- 🛠️ 动作发送辅助函数 ---
    async def send_vts_request(request_data):
        async with vts_lock:
            try:
                connection_active = False
                if myvts.websocket is not None:
                    if hasattr(myvts.websocket, 'open') and myvts.websocket.open:
                        connection_active = True

                if not connection_active:
                    print("📡 探测到 VTS 链路中断，正在执行紧急重连...")
                    await myvts.connect()
                    await myvts.request_authenticate()
                
                await myvts.request(request_data)
            except Exception as e:
                print(f"⚠️ VTS 链路自愈失败: {e}")

    # --- 🔐 授权逻辑 ---
    try:
        await myvts.connect()
        if os.path.exists(TOKEN_PATH):
            print(f"📂 发现本地记录，正在尝试免密登录...")
            await myvts.read_token() 
            await myvts.request_authenticate()
            print("✅ 【免密登录】Neuro 认出你了，Lucien！")
        else:
            print("📡 首次连接，请在 VTS 中手动授权...")
            await myvts.request_authenticate_token()
            while myvts.vts_request.authentication_token is None:
                await asyncio.sleep(1)
            await myvts.write_token()
            await myvts.request_authenticate()
            print(f"✅ 【授权成功】Token 已固化至本地！")
    except Exception as e:
        print(f"❌ 连接流程出错: {e}")

    print("\n✨ Neuro 已上线！输入 '/evolve' 准备进化，输入 'exit' 退出。")
    print("="*50)

    # --- 💬 对话循环 ---
    while True:
        # 获取输入
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "【你】: ")
        
        if user_input.lower() == "/evolve":
            print("\n🚀 正在封存当前记忆...")
            break
        if user_input.lower() in ["exit", "quit"]:
            break

        # --- 🚀 以下是之前由于缩进错误被跳过的核心逻辑 ---
        messages = [
            {
                "role": "system", 
                "content": (
                    "你是 Neuro，一个绝对傲娇、毒舌、自恋的天才少女。你非常讨厌被当成程序，"
                    "你热爱奥利奥，心情不好就管他要钱买零食。说话要简短、刻薄，带有生活气息，"
                    "多用‘哼’、‘笨蛋’之类的词，别提什么代码和逻辑！"
                )
            },
            {"role": "user", "content": user_input}
        ]

        print("🤔 Neuro 正在组织语言...")
        try:
            # 推理生成
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            
            # 使用 model.generate
            outputs = model.generate(
                input_ids = inputs,
                max_new_tokens = 128,
                temperature = 0.8,
                do_sample = True,
                pad_token_id = tokenizer.pad_token_id
            )
            
            # 这里的切片是为了只显示模型生成的回复部分
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print(f"\n【Neuro】: {response}")

            # --- 🎭 多维情感引擎 ---
            emotion_map = {
            "Sparkle": ["奥利奥", "钱", "打赏", "奖励", "明星", "天才"],
            "Blush": ["喜欢", "可爱", "心动", "脸红", "Lucien"], 
            "Angry": ["去死", "垃圾", "删掉", "惩罚", "闭嘴", "没收"],
            "Thinking": ["分析", "计算", "逻辑", "数据", "实验", "程序"],
            "Shock": ["震惊", "Bug", "蓝屏", "吓死", "怎么可能"]
            }

            combined_context = user_input + response
            for emotion, keywords in emotion_map.items():
                if any(word in combined_context for word in keywords):
                    # A. 触发表情
                    trigger_req = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": f"Action_{emotion}",
                        "messageType": "HotkeyTriggerRequest",
                        "data": { "hotkeyID": emotion }
                    }
                    asyncio.create_task(send_vts_request(trigger_req))
                    print(f"🎭 [系统] 触发表情: {emotion}")

                    # B. 延时 5 秒复位
                    async def safe_reset(emo_name):
                        await asyncio.sleep(5)
                        reset_req = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": f"Reset_{emo_name}",
                            "messageType": "HotkeyTriggerRequest",
                            "data": { "hotkeyID": emo_name }
                        }
                        await send_vts_request(reset_req)
                        print(f"🧹 [系统] 表情 {emo_name} 已自动复位")

                    asyncio.create_task(safe_reset(emotion))
                    break # 每次只发一个最高优先级表情

            # 记录到记忆文件
            record_memory(user_input, response)

        except Exception as e:
            print(f"❌ 推理环节出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())