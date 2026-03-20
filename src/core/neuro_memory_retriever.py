import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 目录统一基于仓库根目录解析，避免工作目录变化导致找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# 1. 初始化轻量级模型 (在 CPU 上运行，不占显存)
print("🧠 正在初始化记忆提取器...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 

def load_memories(file_path):
    memories = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 存储格式：[指令, 回答]
                memories.append(f"用户说: {data['instruction']} | Neuro答: {data['output']}")
    return memories

# 2. 建立索引
def build_index(memories):
    print(f"📚 正在对 {len(memories)} 条历史记录进行索引...")
    # 将文本转为向量
    embeddings = embed_model.encode(memories)
    dimension = embeddings.shape[1]
    
    # 使用 FAISS 建立向量库
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

# 3. 搜索函数
def search_memory(query, memories, index, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), top_k)
    
    results = [memories[i] for i in indices[0] if i != -1]
    return results

# --- 测试运行 ---
history_candidates = [
    os.path.join(DATA_DIR, "history_growth.jsonl"),
    os.path.join(REPO_ROOT, "history_growth.jsonl"),
]
history_file = next((p for p in history_candidates if os.path.exists(p)), history_candidates[0])

if os.path.exists(history_file):
    all_memories = load_memories(history_file)
    memory_index = build_index(all_memories)
    
    # 模拟查询
    test_query = "我昨天提到了什么零食？"
    related_docs = search_memory(test_query, all_memories, memory_index)
    
    print("\n🔍 检索到的相关记忆：")
    for doc in related_docs:
        print(f" - {doc}")
else:
    print("⚠️ 找不到 history_growth.jsonl，请先确认文件路径。")