import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

print("[mem] loading embedding model...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_memories(file_path):
    memories = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                memories.append(f"user: {data['instruction']} | Neuro: {data['output']}")
    return memories

def build_index(memories):
    print(f"[mem] indexing {len(memories)} records...")
    embeddings = embed_model.encode(memories)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_memory(query, memories, index, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), top_k)
    return [memories[i] for i in indices[0] if i != -1]

# --- Test ---
history_candidates = [
    os.path.join(DATA_DIR, "history_growth.jsonl"),
    os.path.join(REPO_ROOT, "history_growth.jsonl"),
]
history_file = next((p for p in history_candidates if os.path.exists(p)), history_candidates[0])

if os.path.exists(history_file):
    all_memories = load_memories(history_file)
    memory_index = build_index(all_memories)
    related = search_memory("我昨天提到了什么零食？", all_memories, memory_index)
    print("[mem] search results:")
    for doc in related:
        print(f"  - {doc}")
else:
    print("[mem] history_growth.jsonl not found")