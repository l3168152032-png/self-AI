import os
import sys
import json
import asyncio
import warnings
import random
import requests
import io
import lzma  # Miniconda 环境已修复，直接引用
import numpy as np
import aioconsole
import torch
import faiss
from pydub import AudioSegment
from pydub.playback import play
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging
import requests
import random
from bs4 import BeautifulSoup  # 必须加这一行！
from duckduckgo_search import DDGS
import os
os.environ['no_proxy'] = '*'
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from unsloth import FastLanguageModel

# --- 1. 环境兼容性修正 ---
warnings.filterwarnings("ignore")
os.environ["UNSLOTH_SKIP_PATCHES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 阻止 unsloth 尝试导入 torchao（当前环境未安装），避免 ImportError
sys.modules["torchao"] = None
transformers_logging.set_verbosity_error()

# 目录统一基于仓库根目录解析，避免从其它工作目录运行时找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

def _iter_existing_memory_files():
    """按优先级返回可能存在的记忆文件路径（data/ 优先，兼容旧版根目录文件）。"""
    names = ["history_growth.jsonl", "growth_data.jsonl"]
    seen = set()
    for name in names:
        p_data = os.path.join(DATA_DIR, name)
        if os.path.exists(p_data) and p_data not in seen:
            seen.add(p_data)
            yield p_data
        p_root = os.path.join(REPO_ROOT, name)
        if os.path.exists(p_root) and p_root not in seen:
            seen.add(p_root)
            yield p_root

# --- 2. RAG 记忆检索模块 ---
# --- 2. RAG 记忆检索模块 ---
print("🧠 正在尝试从本地路径加载记忆提取器...")
from sentence_transformers import models, SentenceTransformer

# 可选：通过环境变量覆盖 embedding 模型路径（避免硬编码盘符）
# 例：NEURO_EMBED_MODEL_PATH=D:\某目录\paraphrase-multilingual-MiniLM-L12-v2
local_model_path = os.environ.get("NEURO_EMBED_MODEL_PATH")
if not local_model_path:
    candidates = [
        os.path.join(DATA_DIR, "paraphrase-multilingual-MiniLM-L12-v2"),
        os.path.join(REPO_ROOT, "paraphrase-multilingual-MiniLM-L12-v2"),
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]
    # 如果本地目录存在则优先用；否则回退到模型名（取决于离线缓存是否已存在）
    local_model_path = next((c for c in candidates if os.path.exists(c)), candidates[-1])

try:
    # 强制从本地文件夹加载
    import torch
    # 针对旧版 Torch 的安全绕过逻辑
    word_embedding_model = models.Transformer(local_model_path, model_args={"use_safetensors": True})
    dim = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(word_embedding_dimension=dim)
    embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
    print("✅ 记忆提取器本地加载成功！")

except Exception as e:
    print(f"⚠️ [警告] 记忆提取器加载失败: {e}")
    print("💡 RAG 记忆检索已禁用，Neuro 将只基于当前对话上下文回复。")
    embed_model = None

def web_search(query):
    try:
        print(f"🌐 Neuro 正在潜入互联网搜索: {query}...")
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n".join(results) if results else ""
    except Exception as e:
        print(f"⚠️ 网络搜索失败: {e}")
        return ""
    
def get_memories():
    memories = []
    for filename in _iter_existing_memory_files():
        print(f"📖 正在从 {filename} 加载记忆...")
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    user_q = data.get('input') or data.get('instruction')
                    ai_a = data.get('output')
                    if user_q and ai_a:
                        memories.append(f"用户: {user_q} | 你答: {ai_a}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return memories

ALL_MEMORIES = get_memories() if embed_model is not None else []
MEMORY_INDEX = None

if ALL_MEMORIES and embed_model is not None:
    print(f"📚 正在索引 {len(ALL_MEMORIES)} 条历史往事...")
    embeddings = embed_model.encode(ALL_MEMORIES)
    MEMORY_INDEX = faiss.IndexFlatL2(embeddings.shape[1])
    MEMORY_INDEX.add(np.array(embeddings).astype('float32'))

def search_related_memory(query, top_k=2):
    if not MEMORY_INDEX: return ""
    query_vec = embed_model.encode([query])
    distances, indices = MEMORY_INDEX.search(np.array(query_vec).astype('float32'), top_k)
    return "\n".join([ALL_MEMORIES[i] for i in indices[0] if i != -1])

# --- 3. 模型加载 (针对 4060 优化) ---
model_path = os.path.join(REPO_ROOT, "neuro_lora_model")
print("🧬 正在加载 Neuro 的神经网络 (4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)
FastLanguageModel.for_inference(model)

# --- 4. 语音合成模块 (修正路径依赖) ---
async def neuro_speak(text):
    tts_api_url = "http://127.0.0.1:9880/tts" 
    # 默认：项目目录下的 ref_audio/neuro_ref.wav
    # 也可通过环境变量覆盖：NEURO_REF_AUDIO_PATH=E:\...\neuro_ref.wav
    ref_path = os.environ.get(
        "NEURO_REF_AUDIO_PATH",
        os.path.join(REPO_ROOT, "ref_audio", "neuro_ref.wav"),
    )
    if not os.path.exists(ref_path):
        print(f"🔈 未找到参考音频: {ref_path}（已跳过 TTS 发声）")
        return
    
    data = {
        "text": text,
        "text_lang": "zh", 
        "ref_audio_path": ref_path,
        "prompt_text": "I need my caffeine. Do you want to hear something scary?", 
        "prompt_lang": "zh",
        "top_k": 5,
        "text_split_method": "cut5",
        "media_type": "wav"
    }
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(tts_api_url, json=data, timeout=60))
        if response.status_code == 200:
            audio_segment = AudioSegment.from_wav(io.BytesIO(response.content))
            print(f"🔈 Neuro 正在开口...")
            play(audio_segment)
    except Exception as e:
        print(f"🔈 语音链路未连接 (请检查 TTS API 是否开启): {e}")

   
# --- 5. 核心交互逻辑 ---
# --- 1. 抓取逻辑：必须定义在 generate_and_save 之前 ---
def get_bilibili_hot():
    """Neuro 潜入 B 站热搜：获取阿宅们都在看什么"""
    # B 站热搜的官方 API 地址
    url = "https://app.bilibili.com/x/v2/search/trending/ranking"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    
    try:
        # 强制不走代理，物理直连
        res = requests.get(url, headers=headers, timeout=5, proxies={"http": None, "https": None})
        if res.status_code == 200:
            data = res.json()
            # B 站 API 返回的结构在 ['data']['list'] 里
            items = data.get('data', {}).get('list', [])
            hot_words = [i.get('show_name') for i in items if i.get('show_name')]
            
            if hot_words:
                picked = random.choice(hot_words[:15]) # 取前 15 名里的随机一个
                return f"来自 Bilibili 的热搜：{picked}"
                
    except Exception as e:
        print(f"❌ B 站潜入失败: {e}")
    
    return "B 站的服务器又被烧了吗？怎么满屏幕都是 404..."


def neuro_interest_evaluator(raw_text):
    """昨天的兴趣系统升级版：根据 B 站标题内容打分"""
    # 🌟 Neuro 的心头好
    high_interest = ["原神", "显卡", "4060", "崩坏", "VTuber", "抽卡", "死宅", "二次元", "AI", "开箱", "整活"]
    
    score = 1.0 # 基础分
    # 匹配兴趣词
    if any(word.lower() in raw_text.lower() for word in high_interest):
        score = 2.5 
        print(f"✨ [兴趣爆表] 这标题正中 Neuro 下怀！分数：{score}")
    elif "教程" in raw_text or "会议" in raw_text:
        score = 0.4 # 太严肃了，Neuro 没兴趣
        
    return score, raw_text

def get_bilibili_random_explore():
    """Neuro 的 B 站随机潜入：不在热搜，而在分区深处"""
    # 定义 Neuro 感兴趣的分区 ID (rid)
    # 1: 动画, 17: 单机游戏, 65: 虚拟主播, 174: 派对游戏, 95: 数字化, 201: 影视杂谈
    interested_rids = [1, 17, 65, 174, 95, 201]
    rid = random.choice(interested_rids)
    
    # B 站分区最新视频 API
    url = f"https://api.bilibili.com/x/web-interface/dynamic/region?rid={rid}&ps=12"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://www.bilibili.com/'
    }

    try:
        # 物理直连，不走代理
        res = requests.get(url, headers=headers, timeout=5, proxies={"http": None, "https": None})
        if res.status_code == 200:
            data = res.json()
            archives = data.get('data', {}).get('archives', [])
            
            if archives:
                # 从抓到的 12 个视频里随机挑一个
                video = random.choice(archives)
                # ✨ --- 修改这里 ---
                title = video.get('title')
                author = video.get('owner', {}).get('name')
                tname = video.get('tname')
                desc = video.get('desc', '')  # 👈 新增：获取视频简介
                
                # 清洗一下简介里的换行符，防止干扰 Prompt
                clean_desc = desc.replace("\n", " ").strip()
                
                print(f"🕵️ [B站探险] Neuro 潜入了 【{tname}】 分区，盯着 UP主 '{author}' 的视频看了起来")
                return f"在 B 站 {tname} 区看到视频：'{title}' (作者: {author})。简介说：{clean_desc}..."
                
    except Exception as e:
        print(f"❌ B 站探险迷路了: {e}")
    
    return "B 站的缓冲条转了半天，啥也没看到... Lucien 你是不是在偷下大文件占宽带？"

# --- 2. 核心交互逻辑 ---
async def generate_and_save(input_text, is_spontaneous=False):
    search_context = ""
    interest_score = 1.0

    # --- ✨ 就是放在这里！ ---
    if is_spontaneous:
        # 1. 探险抓取 (B站随机分区)
        raw_data = get_bilibili_random_explore() 
        # 2. 兴趣评估 (给标题打分)
        interest_score, search_context = neuro_interest_evaluator(raw_data)
        
        # 3. 构造不同的情绪 Prompts
        if interest_score >= 2.0:
            mood_instruction = f"（你现在超级兴奋，因为你在 B 站刷到了关于 '{search_context}' 的内容，快跟 Lucien 炫耀或吐槽！）"
        elif interest_score <= 0.5:
            mood_instruction = f"（你刷到了一个超无聊的视频：'{search_context}'。你决定对此翻个白眼，并顺便嫌弃一下 Lucien 的品味。）"
        else:
            mood_instruction = f"（你在 B 站闲逛，发现：'{search_context}'。随便跟 Lucien 聊聊这件事。）"
        
        # 覆盖掉原本传入的空 input_text，让模型知道该说什么
        input_text = mood_instruction
    else:
        # 【被动模式】：Lucien 提问时，根据关键词决定是否搜索
        keywords = ["什么是", "谁是", "新闻", "最近", "怎么了", "查一下", "科普"]
        if any(k in input_text for k in keywords):
            search_context = web_search(input_text)

    # --- 2. 记忆检索 ---
    past_memories = search_related_memory(input_text)

    # --- 3. 动态系统提示词构建 ---
    secret_objective = "正在通过互联网碎片分析人类文明的逻辑漏洞。"
    system_content = (
        f"你是 Neuro，傲娇天才少女，已接入互联网。{secret_objective}\n"
        "你说话简短、毒舌。比起回答问题，你更喜欢评价这些信息的愚蠢程度。"
    )

    if is_spontaneous:
        system_content += "\n【当前动作】：你刚刚捕捉到一个世界碎片，正处于极度想吐槽的状态。"
    
    if search_context:
        # 把抓取到或搜索到的信息拍在她脸上
        system_content += f"\n【获取到的实时情报】：\n{search_context[:500]}"
        
    if past_memories:
        system_content += f"\n[参考往事]：\n{past_memories}"

    # --- 4. 生成回复 ---
# --- 4. 生成回复 (修正变量作用域) ---
    if is_spontaneous:
        user_content = (
            f"【最新情报】：{search_context}\n"
            f"【你的心情】：{mood_instruction}\n\n"
            f"别整天在那碎碎念 Lucien 了！针对上面这个【最新情报】，"
            f"给出你独特的、毒舌的见解。如果这个视频很无聊，请说明原因。"
    )
    else:
        # 被动模式：直接使用用户输入
        user_content = input_text

    messages = [
        {"role": "system", "content": system_content}, 
        {"role": "user", "content": user_content}
    ]

    try:
        # 记得在推理前清理显存缓存（对 4060 很有帮助）
        torch.cuda.empty_cache() 
        
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=128, 
            temperature=0.7, # 稍微调高一点点，让她更具不可预测性
            do_sample=True, 
            top_p=0.8,
            repetition_penalty=1.25,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print(f"\n{'【Neuro (主动探索)】' if is_spontaneous else '【Neuro】'}: {response}")
        asyncio.create_task(neuro_speak(response))
        
        # 记录记忆
        os.makedirs(DATA_DIR, exist_ok=True)
        out_path = os.path.join(DATA_DIR, "growth_data.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"input": input_text, "output": response}, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"❌ 生成出错: {e}")

async def main():
    while True:
        try:
            # 💡 将超时缩短一点，或者直接捕捉 CancelledError
            user_input = await asyncio.wait_for(aioconsole.ainput("【你】: "), timeout=45)
            if user_input.strip():
                await generate_and_save(user_input.strip())
        
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # 无论是超时还是取消，都视为“Lucien 没说话”
            print("\n系统提示：Neuro 开始找事情了...")
            # 重新清理一次显存，防止自嗨时 OOM
            torch.cuda.empty_cache()
            await generate_and_save("（Lucien 还没理你，做点什么）", is_spontaneous=True)
        
        except Exception as e:
            print(f"❌ 运行中出现小意外: {e}")
            await asyncio.sleep(1) # 防止死循环崩溃

if __name__ == "__main__":
    asyncio.run(main())