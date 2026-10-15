# Mini Transformer / 手写 Transformer

一个用于学习 Transformer 内部机制的教学型 Mini GPT 项目。项目从张量运算、QKV、因果掩码和多头注意力开始，逐步组合成 Transformer block，最后完成一个极小的字符级语言模型训练与自回归生成流程。

> This is a small, from-scratch learning project for understanding the internal mechanics of a Transformer. It builds the model step by step—from tensor operations, QKV projections, causal masking, and multi-head attention to a complete character-level Mini GPT training and generation loop.

## 项目目标 / Goals

- 用尽量少的抽象观察 Transformer 的核心张量形状和数据流。
- 理解 self-attention、causal mask、residual connection、LayerNorm、FFN 和 next-token prediction。
- 用一个极小的中文字符语料跑通训练、保存 checkpoint 和自回归生成。

The goal is understanding and experimentation, not competitive language modeling. The training corpus is intentionally tiny, so the model mainly learns to reproduce local patterns from the demo sentences.

## 结构总览 / Project map

```text
01_tensor.py          基础张量与线性变换 / tensor and linear algebra basics
02_qkv.py             Q/K/V 投影与注意力分数 / QKV projections and scores
03_train_qkv.py       对 QKV 参数做最小训练实验 / minimal QKV training experiment
04_causal_mask.py     因果掩码：当前位置不能看未来 / causal attention mask
05_multi_head.py      多头注意力的拆分与合并 / split and merge attention heads
06_transformer_block.py
                      Transformer block：Norm → Attention → Residual → FFN
07_embedding.py       token embedding + positional embedding
08_mini_gpt.py        从 embedding 到 logits 的单文件教学演示
09_train.py           字符级 next-token 训练并保存 checkpoint
10_generate.py        加载 checkpoint，逐 token 自回归生成
mini_gpt_model.py     可复用的 MiniGPT、Attention、FFN 和 Block 定义
gpu_test.py           CUDA 环境检查
```

## 数据流 / Model path

```text
字符文本
  ↓
字符词表（stoi / itos）
  ↓
token ids: (B, T)
  ↓
token embedding + positional embedding
  ↓
重复 num_layers 次 TransformerBlock
  ├─ LayerNorm
  ├─ causal multi-head self-attention
  ├─ residual connection
  ├─ LayerNorm
  ├─ GELU feed-forward network
  └─ residual connection
  ↓
final LayerNorm + language-model head
  ↓
logits: (B, T, vocab_size)
  ↓
next-character cross entropy loss / autoregressive generation
```

在当前默认配置中，`09_train.py` 使用 `d_model=32`、`num_heads=4`、`hidden_dim=128`、`num_layers=2`、`context_length=16`。这些参数是为了便于观察和快速实验，不代表生产模型配置。

## 环境 / Requirements

- Python `>=3.12`
- PyTorch `>=2.14.1`
- NumPy `>=2.5.2`
- 有 CUDA 时脚本会自动使用 GPU，否则回退到 CPU。

The lock file was generated with `uv`. The project also contains a CUDA 12.6 PyTorch index configuration; if you do not need CUDA, install a CPU-compatible PyTorch build according to your platform.

## 安装 / Installation

使用 `uv`：

```bash
uv sync
```

也可以使用普通 Python 环境：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install numpy torch
```

## 运行 / Run

按学习顺序运行前面的实验：

```bash
python 01_tensor.py
python 02_qkv.py
python 04_causal_mask.py
python 05_multi_head.py
python 06_transformer_block.py
python 07_embedding.py
python 08_mini_gpt.py
```

训练字符级 Mini GPT：

```bash
python 09_train.py
```

训练完成后会在本地生成 `checkpoints/mini_gpt.pt`。训练步数可以通过环境变量调整：

```bash
STEPS=600 python 09_train.py       # Linux/macOS
$env:STEPS=600; python 09_train.py # PowerShell
```

加载 checkpoint 并生成：

```bash
python 10_generate.py --prompt "我喜欢" --tokens 24
python 10_generate.py --prompt "我喜欢" --tokens 24 --sample --temperature 0.8 --top-k 3
```

`10_generate.py` 的 prompt 只能使用训练词表中的字符。由于语料很小，生成结果主要用于验证流程，不应视为有实际语言能力的模型。

## Checkpoints / 模型文件

模型 checkpoint 是训练产生的二进制输出，默认不会提交到 Git。克隆仓库后先运行 `09_train.py`，再运行 `10_generate.py`。这样可以避免把本地实验产物混入源码历史，也让复现实验路径更明确。

The repository intentionally ignores generated checkpoints. Run the training script first to create the local file required by the generation script.

## 学习记录 / Learning notes

这个仓库记录的是我对 Transformer 原理的逐步拆解和实验实现。代码优先服务于可读性：部分矩阵参数直接使用 `nn.Parameter`，没有引入完整生产框架中的 tokenizer、数据管道、分布式训练或复杂配置系统。

This repository is a hands-on implementation for learning. It favors inspectable tensor operations over production abstractions, and intentionally omits a full tokenizer, large-scale data pipeline, distributed training, and production configuration management.

## License

当前仓库尚未选择开源许可证。如果希望别人明确知道如何使用、修改和分发代码，建议后续添加 MIT 或 Apache-2.0 等许可证。
