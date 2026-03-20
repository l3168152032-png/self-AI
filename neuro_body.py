import asyncio
import pyvts
import json
import os

# 1. 配置表情映射
EMOTION_MAP = {
    "Angry": ["去死", "垃圾", "删掉", "惩罚", "闭嘴", "笨蛋", "气死", "不准吃"],
    "Blush": ["喜欢", "可爱", "心动", "害羞", "Lucien", "表现不错"],
    "Sparkle": ["奥利奥", "钱", "打赏", "奖励", "天才", "聪明", "想啊", "Offer", "拿来"],
}

active_emotions = set() 

async def trigger_vts(myvts, emotion):
    if emotion in active_emotions: return 
    active_emotions.add(emotion)
    try:
        # 1. 触发表情/动画
        req = {
            "apiName": "VTubeStudioPublicAPI", 
            "apiVersion": "1.0", 
            "requestID": f"Action_{emotion}", 
            "messageType": "HotkeyTriggerRequest",
            "data": { "hotkeyID": emotion }
        }
        await myvts.request(req)
        print(f"🎭 [身体动态] 确认触发: {emotion}")
        
        # 等待 5 秒展示时间
        await asyncio.sleep(5)
        
        # 2. 复位该表情
        req["requestID"] = f"Reset_{emotion}"
        await myvts.request(req)

        await asyncio.sleep(0.5)
        
        # 3. 💡 新增：立即触发“嘴部恢复”热键
        # 请确保引号里的名字和 VTS 热键列表里的 Item Name 完全一致
        mouth_fix_req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "Fix_Mouth_State",
            "messageType": "HotkeyTriggerRequest",
            "data": { "hotkeyID": "Mouth_Reset" } 
        }
        await myvts.request(mouth_fix_req)
        
        print(f"🧹 [身体动态] 复位完毕: {emotion}，嘴部状态已强制校准。")
        
    finally:
        active_emotions.remove(emotion)

async def watch_logic():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TOKEN_PATH = os.path.join(BASE_DIR, "pyvts_token.txt")
    # 兼容：优先监听 data/ 下的增长文件，回退到旧版根目录文件
    MEMORY_FILE = os.path.join(DATA_DIR, "growth_data.jsonl")
    if not os.path.exists(MEMORY_FILE):
        MEMORY_FILE = os.path.join(BASE_DIR, "growth_data.jsonl")
    
    myvts = pyvts.vts(port=8001)
    myvts.vts_request = pyvts.vts_request.VTSRequest(
        plugin_name="Neuro_Body_Control",
        developer="Lucien",
        authentication_token_path=TOKEN_PATH
    )

    await myvts.connect()
    
    # --- 1. 授权验证环节 ---
    authorized = False
    if os.path.exists(TOKEN_PATH):
        await myvts.read_token()
        auth_resp = await myvts.request_authenticate()
        if isinstance(auth_resp, bool):
            authorized = auth_resp
        elif isinstance(auth_resp, dict):
            authorized = auth_resp.get('data', {}).get('authenticated', False)

    if not authorized:
        print("📡 [Body] 请在 VTS 窗口中点击【允许】以授权新插件...")
        await myvts.request_authenticate_token()
        while myvts.vts_request.authentication_token is None:
            await asyncio.sleep(1)
        await myvts.write_token()
        await myvts.request_authenticate()
        print("✅ [Body] 授权成功！")

    # --- 2. 核心监听环节 (确保这部分在 async 函数内部) ---
    print("👀 [系统] 开始监听 Neuro 的发言记录...")
    if not os.path.exists(MEMORY_FILE):
        open(MEMORY_FILE, 'w').close()
    
    last_size = os.path.getsize(MEMORY_FILE)

    while True:
        try:
            current_size = os.path.getsize(MEMORY_FILE)
            if current_size > last_size:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    f.seek(last_size)
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        
                        user_q = data.get("instruction", "")
                        ai_a = data.get("output", "")
                        triggered = False

                        # A. 优先级最高：用户输入检测
                        # 如果你在骂她，优先触发 Angry，无视她的狡辩
                        if any(word in user_q for word in ["笨", "菜", "Bug", "垃圾", "慢", "笨蛋"]):
                            asyncio.create_task(trigger_vts(myvts, "Angry"))
                            triggered = True

                        # B. 优先级次之：AI 回复内容匹配
                        if not triggered:
                            # 按照设定的情绪权重顺序匹配
                            for emotion in ["Angry", "Shock", "Sparkle", "Blush"]:
                                keywords = EMOTION_MAP.get(emotion, [])
                                if any(word in ai_a for word in keywords):
                                    asyncio.create_task(trigger_vts(myvts, emotion))
                                    triggered = True
                                    break 
                last_size = current_size
        except Exception as e:
            print(f"⚠️ 监听循环波动: {e}")
        
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    try:
        asyncio.run(watch_logic())
    except KeyboardInterrupt:
        print("\n👋 身体控制脚本已关闭")