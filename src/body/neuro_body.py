import asyncio
import pyvts
import json
import os

EMOTION_MAP = {
    "Angry": ["去死", "垃圾", "删掉", "惩罚", "闭嘴", "笨蛋", "气死", "不准吃"],
    "Blush": ["喜欢", "可爱", "心动", "害羞", "Lucien", "表现不错"],
    "Sparkle": ["奥利奥", "钱", "打赏", "奖励", "天才", "聪明", "想啊", "Offer", "拿来"],
}

active_emotions = set()

async def trigger_vts(myvts, emotion):
    if emotion in active_emotions:
        return
    active_emotions.add(emotion)
    try:
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"Action_{emotion}",
            "messageType": "HotkeyTriggerRequest",
            "data": {"hotkeyID": emotion}
        }
        await myvts.request(req)
        print(f"[vts] triggered: {emotion}")
        await asyncio.sleep(5)
        req["requestID"] = f"Reset_{emotion}"
        await myvts.request(req)
        await asyncio.sleep(0.5)
        # Force mouth reset after each emotion to fix lip-sync state
        await myvts.request({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "Fix_Mouth_State",
            "messageType": "HotkeyTriggerRequest",
            "data": {"hotkeyID": "Mouth_Reset"}
        })
        print(f"[vts] reset: {emotion}")
    finally:
        active_emotions.remove(emotion)

async def watch_logic():
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    DATA_DIR = os.path.join(REPO_ROOT, "data")
    TOKEN_PATH = os.path.join(REPO_ROOT, "pyvts_token.txt")
    MEMORY_FILE = os.path.join(DATA_DIR, "growth_data.jsonl")
    if not os.path.exists(MEMORY_FILE):
        MEMORY_FILE = os.path.join(REPO_ROOT, "growth_data.jsonl")

    myvts = pyvts.vts(port=8001)
    myvts.vts_request = pyvts.vts_request.VTSRequest(
        plugin_name="Neuro_Body_Control",
        developer="Lucien",
        authentication_token_path=TOKEN_PATH
    )

    await myvts.connect()

    # --- Auth ---
    authorized = False
    if os.path.exists(TOKEN_PATH):
        await myvts.read_token()
        auth_resp = await myvts.request_authenticate()
        if isinstance(auth_resp, bool):
            authorized = auth_resp
        elif isinstance(auth_resp, dict):
            authorized = auth_resp.get('data', {}).get('authenticated', False)

    if not authorized:
        print("[vts] waiting for VTS authorization...")
        await myvts.request_authenticate_token()
        while myvts.vts_request.authentication_token is None:
            await asyncio.sleep(1)
        await myvts.write_token()
        await myvts.request_authenticate()
        print("[vts] authorized")

    print("[vts] watching for Neuro responses...")
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
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        user_q = data.get("instruction", "")
                        ai_a = data.get("output", "")
                        triggered = False

                        # User anger detection has highest priority
                        if any(word in user_q for word in ["笨", "菜", "Bug", "垃圾", "慢", "笨蛋"]):
                            asyncio.create_task(trigger_vts(myvts, "Angry"))
                            triggered = True

                        if not triggered:
                            for emotion in ["Angry", "Shock", "Sparkle", "Blush"]:
                                keywords = EMOTION_MAP.get(emotion, [])
                                if any(word in ai_a for word in keywords):
                                    asyncio.create_task(trigger_vts(myvts, emotion))
                                    break
                last_size = current_size
        except Exception as e:
            print(f"[vts] error: {e}")

        await asyncio.sleep(0.5)

if __name__ == "__main__":
    try:
        asyncio.run(watch_logic())
    except KeyboardInterrupt:
        print("[vts] shutdown")