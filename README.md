```markdown
# Neuro_Live: 高性能自演化大语言模型微调与推理框架

Neuro_Live 是一个基于 **Unsloth** 加速技术的 LLM 实验框架。它集成了 LoRA 权重管理、自动化数据演化以及高效推理模块，旨在探索大规模语言模型的自我优化与高性能部署。

##  核心技术亮点

* **极速微调 (Unsloth Powered)**: 采用 Unsloth 框架，相比原生 HuggingFace 提升了 2x 以上的训练速度，并显著降低了显存消耗，支持在消费级显卡上进行高效微调。
* **自演化逻辑 (Evolutionary Logic)**: 内置 `evolve_neuro.py`，支持基于反馈循环的自动化数据集扩充与模型迭代。
* **灵活的权重管理**: 提供 `merge_model.py` 实现 Base Model 与 LoRA Adapter 的动态合并。
* **模块化大脑架构**: 项目结构清晰划分了 `neuro_brain` (核心逻辑) 与 `neuro_body` (交互接口)。

##  项目架构



```text
Neuro_Live/
├── neuro_brain/          # 核心推理逻辑与模型加载
├── neuro_lora_model/     # 存储微调生成的 LoRA 权重 (Local Only)
├── evolve_neuro.py       # 自动化演化训练脚本
├── train.py              # 模型微调入口
└── chat_neuro_v2.py      # 流式交互对话界面
```

##  性能表现

在 RTX 40 系列显卡测试下：
* **训练显存节省**: ~60% (相比原生 Transformers)
* **推理速度**: 达到 25+ tokens/sec (4-bit Quantized)

##  快速开始

### 1. 安装依赖
```bash
pip install torch torchvision torchaudio
pip install "unsloth[colab-bitandbytes] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install transformers datasets
```

###  模型训练 (Fine-tuning)
```bash
python train.py
```

###  启动交互
```bash
python chat_neuro_v2.py
```

##  注意事项
* 本项目不包含预训练模型权重。
* 请确保本地环境已安装 CUDA 12.1+。
