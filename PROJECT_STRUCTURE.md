# 仓库文件分类（按用途）

> 说明：本文件仅用于“整理分类”，不改变代码结构或导入路径。

## 1) 推理 / 对话入口（Neuro 运行）
- `start_neuro.py`：启动 Neuro 的主入口（通常用于开启主体逻辑）
- `chat_neuro.py`：对话式交互入口（早期/简版）
- `chat_neuro_v2.py`：对话式交互入口（v2，通常包含更多能力/逻辑）
- `neuro_body.py`：VTS（身体/表情）控制与联动监听
- `interact.py`：交互辅助脚本（与身体/热键/流程联动相关）
- `test_neuro.py`：对 Neuro 逻辑做基本测试

## 2) 核心大脑（模型加载 + 生成逻辑）
- `neuro_brain.py`：核心推理与生成逻辑（系统提示词、检索记忆、调用 TTS 等）
- `neuro_memory_retriever.py`：记忆检索模块（embedding + FAISS）

## 3) 记忆数据 / 训练数据（JSONL）
- `history_growth.jsonl`：历史累计记忆（归档后长期使用）
- `growth_data_v2.jsonl`：临时增长记忆（用于训练/进化）
- `neuro_train.jsonl`：训练数据样例/配置用 JSONL

## 4) 训练与进化（LoRA / 自演化）
- `train.py`：训练入口（SFT/微调相关）
- `evolve_neuro.py`：进化入口（把临时增长记忆合并/再训练）
- `dataset.py`：数据集构建/读取逻辑
- `merge_model.py`：合并 Base 模型与 LoRA Adapter 的脚本
- `seed_data.py`：种子数据生成/初始化
- `manual_save.py`：手动保存/归档相关脚本
- `neuro_identity.json`：Neuro 身份/设定（用于提示词或角色描述）

## 5) 配置与开发文件
- `.gitignore`：忽略模型权重/缓存等大文件
- `.vscode/`：VSCode 工作区配置

## 6) 建议的目录（可选，未来优化）
如果你准备做“更工程化”的目录结构，建议新增（不强制）：
- `data/`：存放 `*.jsonl`
- `ref_audio/`：存放 `neuro_ref.wav`
- `models/`：存放模型/LoRA（通常不建议直接 push 大文件）

