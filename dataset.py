import json
import re
import os

def clean_neuro_speech(text):
    # 移除语气词前缀和括号内的动作/内心独白
    text = re.sub(r"^(啧|哼|喂|……|啧……|哼……|喂……|嘘|哈|啧。|哼。|喂。)\。?(\s*)", "", text)
    text = re.sub(r"[\(（].*?[\)）]", "", text) # 这一行会直接杀掉所有 (内心独白) 和 (动作)
    
    # 降低攻击性词汇
    insults = ["笨蛋", "小笨蛋", "笨拙的人类", "loser", "弱智", "低智", "小破孩", "蠢货", "傻笑"]
    for word in insults:
        text = text.replace(word, "Lucien")
    
    # 清理多余的空格和换行
    text = text.replace("\n", " ").strip()
    return text

# --- 自动处理 ---
current_dir = os.getcwd()
jsonl_files = [f for f in os.listdir(current_dir) if f.endswith('.jsonl') and '理性版' not in f]

if not jsonl_files:
    print(f"❌ 错误：在 {current_dir} 没找到任何 .jsonl 文件！")
else:
    input_file = jsonl_files[0]
    output_file = "b站语料_理性版.jsonl"
    print(f"🔍 正在清洗: {input_file} ...")

    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            
            try:
                # 尝试标准解析
                data = json.loads(line)
                data['output'] = clean_neuro_speech(data['output'])
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
                processed_count += 1
            except json.JSONDecodeError:
                # 如果一行里挤了多个对象，尝试暴力拆解（针对你贴给我的那种格式）
                parts = re.findall(r'\{.*?\}', line)
                for p in parts:
                    try:
                        data = json.loads(p)
                        data['output'] = clean_neuro_speech(data['output'])
                        out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        processed_count += 1
                    except:
                        continue
                print(f"⚠️ 第 {i} 行格式异常，已尝试自动拆解。")

    print(f"✅ 处理完成！共转换 {processed_count} 条理性语料。")