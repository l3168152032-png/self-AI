import json
import re
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

def clean_neuro_speech(text):
    text = re.sub(r"^(啧|哼|喂|……|啧……|哼……|喂……|嘘|哈|啧。|哼。|喂。)\.?(\s*)", "", text)
    text = re.sub(r"[\(（].*?[\)）]", "", text)
    insults = ["笨蛋", "小笨蛋", "笨拙的人类", "loser", "弱智", "低智", "小破孩", "蠢货", "傻笑"]
    for word in insults:
        text = text.replace(word, "Lucien")
    text = text.replace("\n", " ").strip()
    return text

jsonl_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl') and '理性版' not in f]

if not jsonl_files:
    print(f"[data] no .jsonl files found in {DATA_DIR}")
else:
    input_file = jsonl_files[0]
    output_file = os.path.join(DATA_DIR, "b站语料_理性版.jsonl")
    print(f"[data] cleaning: {input_file} ...")

    processed = 0
    with open(os.path.join(DATA_DIR, input_file), 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data['output'] = clean_neuro_speech(data['output'])
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
                processed += 1
            except json.JSONDecodeError:
                parts = re.findall(r'\{.*?\}', line)
                for p in parts:
                    try:
                        data = json.loads(p)
                        data['output'] = clean_neuro_speech(data['output'])
                        out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        processed += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
                print(f"[data] line {i}: fixed malformed JSON")

    print(f"[data] done: {processed} records cleaned")