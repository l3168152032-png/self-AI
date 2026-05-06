import os, sys, json, io, asyncio, warnings, random
import lzma
import numpy as np
import aioconsole
import torch
import faiss
import requests
from pydub import AudioSegment
from pydub.playback import play
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, models
from transformers import logging as transformers_logging
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

os.environ['no_proxy'] = '*'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

warnings.filterwarnings("ignore")
os.environ["UNSLOTH_SKIP_PATCHES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# unsloth 在无 torchao 环境下会 ImportError，预置空模块绕过
sys.modules["torchao"] = None
transformers_logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

def _iter_existing_memory_files():
    """按优先级返回 data/ 下及根目录的记忆文件路径。"""
    seen = set()
    for name in ("history_growth.jsonl", "growth_data.jsonl"):
        for base in (DATA_DIR, REPO_ROOT):
            p = os.path.join(base, name)
            if os.path.exists(p) and p not in seen:
                seen.add(p)
                yield p

# --- RAG embedding 模型加载 ---
print("[RAG] loading embedding model...")
local_model_path = os.environ.get("NEURO_EMBED_MODEL_PATH")
if not local_model_path:
    candidates = [
        os.path.join(DATA_DIR, "paraphrase-multilingual-MiniLM-L12-v2"),
        os.path.join(REPO_ROOT, "paraphrase-multilingual-MiniLM-L12-v2"),
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]
    local_model_path = next((c for c in candidates if os.path.exists(c)), candidates[-1])

try:
    word_embedding_model = models.Transformer(local_model_path, model_args={"use_safetensors": True})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
    print("[RAG] embedding model loaded")
except Exception as e:
    print(f"[RAG] embedding model failed, RAG disabled: {e}")
    embed_model = None

def web_search(query):
    try:
        print(f"[web] searching: {query[:50]}...")
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n".join(results) if results else ""
    except Exception as e:
        print(f"[web] search failed: {e}")
        return ""

def get_memories():
    memories = []
    for filename in _iter_existing_memory_files():
        print(f"[mem] loading {os.path.basename(filename)}...")
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
    print(f"[RAG] indexing {len(ALL_MEMORIES)} memories...")
    embeddings = embed_model.encode(ALL_MEMORIES)
    MEMORY_INDEX = faiss.IndexFlatL2(embeddings.shape[1])
    MEMORY_INDEX.add(np.array(embeddings).astype('float32'))

def search_related_memory(query, top_k=2):
    if not MEMORY_INDEX: return ""
    query_vec = embed_model.encode([query])
    distances, indices = MEMORY_INDEX.search(np.array(query_vec).astype('float32'), top_k)
    return "\n".join([ALL_MEMORIES[i] for i in indices[0] if i != -1])

# --- 模型加载 ---
model_path = os.path.join(REPO_ROOT, "neuro_lora_model")
print("[model] loading Qwen2.5-7B (4-bit) + LoRA...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)
FastLanguageModel.for_inference(model)
print("[model] ready")

# --- TTS ---
async def neuro_speak(text):
    tts_url = "http://127.0.0.1:9880/tts"
    ref_path = os.environ.get("NEURO_REF_AUDIO_PATH",
                               os.path.join(REPO_ROOT, "ref_audio", "neuro_ref.wav"))
    if not os.path.exists(ref_path):
        print(f"[tts] ref audio not found at {ref_path}, skipped")
        return

    data = {
        "text": text, "text_lang": "zh",
        "ref_audio_path": ref_path,
        "prompt_text": "I need my caffeine. Do you want to hear something scary?",
        "prompt_lang": "zh",
        "top_k": 5, "text_split_method": "cut5", "media_type": "wav"
    }
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: requests.post(tts_url, json=data, timeout=60))
        if resp.status_code == 200:
            play(AudioSegment.from_wav(io.BytesIO(resp.content)))
    except Exception as e:
        print(f"[tts] failed: {e}")

# --- Bilibili content sources ---
_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
_NO_PROXY = {"http": None, "https": None}

def get_bilibili_hot():
    url = "https://app.bilibili.com/x/v2/search/trending/ranking"
    try:
        res = requests.get(url, headers={'User-Agent': _USER_AGENT}, timeout=5, proxies=_NO_PROXY)
        if res.status_code == 200:
            items = res.json().get('data', {}).get('list', [])
            hot_words = [i.get('show_name') for i in items if i.get('show_name')]
            if hot_words:
                return f"Bilibili 热搜: {random.choice(hot_words[:15])}"
    except Exception as e:
        print(f"[bili] hot search failed: {e}")
    return ""


def neuro_interest_evaluator(raw_text):
    high = ["原神", "显卡", "4060", "崩坏", "VTuber", "抽卡", "死宅", "二次元", "AI", "开箱", "整活"]
    if any(w.lower() in raw_text.lower() for w in high):
        return 2.5, raw_text
    if "教程" in raw_text or "会议" in raw_text:
        return 0.4, raw_text
    return 1.0, raw_text

def get_bilibili_random_explore():
    rids = [1, 17, 65, 174, 95, 201]  # 动画, 单机游戏, 虚拟主播, 派对, 数字化, 影视
    rid = random.choice(rids)
    url = f"https://api.bilibili.com/x/web-interface/dynamic/region?rid={rid}&ps=12"
    headers = {'User-Agent': _USER_AGENT, 'Referer': 'https://www.bilibili.com/'}
    try:
        res = requests.get(url, headers=headers, timeout=5, proxies=_NO_PROXY)
        if res.status_code == 200:
            archives = res.json().get('data', {}).get('archives', [])
            if archives:
                v = random.choice(archives)
                tname = v.get('tname', '')
                title = v.get('title', '')
                author = v.get('owner', {}).get('name', '')
                desc = v.get('desc', '').replace("\n", " ").strip()
                print(f"[bili] explore {tname}: '{title}' by {author}")
                return f"B站 {tname} 区: '{title}' (作者: {author})。{desc}"
    except Exception as e:
        print(f"[bili] explore failed: {e}")
    return ""

# --- Core generation ---
async def generate_and_save(input_text, is_spontaneous=False):
    search_context = ""
    interest_score = 1.0

    if is_spontaneous:
        raw_data = get_bilibili_random_explore()
        interest_score, search_context = neuro_interest_evaluator(raw_data)
        if interest_score >= 2.0:
            mood = f"（你超级兴奋，在B站刷到了 '{search_context}'，快跟 Lucien 炫耀或吐槽！）"
        elif interest_score <= 0.5:
            mood = f"（你刷到一个超无聊的视频：'{search_context}'，翻个白眼并嫌弃一下 Lucien 的品味。）"
        else:
            mood = f"（你在B站闲逛发现：'{search_context}'，随便跟 Lucien 聊聊。）"
        input_text = mood
    else:
        if any(k in input_text for k in ["什么是", "谁是", "新闻", "最近", "怎么了", "查一下", "科普"]):
            search_context = web_search(input_text)

    past_memories = search_related_memory(input_text)

    system_content = (
        "你是 Neuro，傲娇天才少女，已接入互联网。正在通过互联网碎片分析人类文明的逻辑漏洞。\n"
        "你说话简短、毒舌。比起回答问题，你更喜欢评价这些信息的愚蠢程度。"
    )
    if is_spontaneous:
        system_content += "\n【当前动作】：你捕捉到一个世界碎片，正处于极度想吐槽的状态。"
    if search_context:
        system_content += f"\n【实时情报】：\n{search_context[:500]}"
    if past_memories:
        system_content += f"\n[参考往事]：\n{past_memories}"

    if is_spontaneous:
        user_content = (
            f"【最新情报】：{search_context}\n"
            f"【你的心情】：{mood}\n\n"
            "别整天碎碎念 Lucien 了！针对上面这个【最新情报】，给出你独特的毒舌见解。如果很无聊，请说明原因。"
        )
    else:
        user_content = input_text

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    try:
        torch.cuda.empty_cache()
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            temperature=0.7, do_sample=True, top_p=0.8,
            repetition_penalty=1.25,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        label = "Neuro (active)" if is_spontaneous else "Neuro"
        print(f"\n【{label}】: {response}")
        asyncio.create_task(neuro_speak(response))

        os.makedirs(DATA_DIR, exist_ok=True)
        out_path = os.path.join(DATA_DIR, "growth_data.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"input": input_text, "output": response}, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[gen] failed: {e}")

async def main():
    while True:
        try:
            user_input = await asyncio.wait_for(aioconsole.ainput("You: "), timeout=45)
            if user_input.strip():
                await generate_and_save(user_input.strip())
        except (asyncio.TimeoutError, asyncio.CancelledError):
            print("\n[system] idle, Neuro exploring...")
            torch.cuda.empty_cache()
            await generate_and_save("", is_spontaneous=True)
        except Exception as e:
            print(f"[system] error: {e}")
            await asyncio.sleep(1)

