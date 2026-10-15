"""实验 10：重新加载训练好的 Mini GPT，逐 Token 自回归生成。"""
import argparse
from pathlib import Path

import torch

from mini_gpt_model import MiniGPT


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="我喜欢", help="必须使用训练词表中存在的字符")
parser.add_argument("--tokens", type=int, default=24, help="新增字符数量")
parser.add_argument("--sample", action="store_true", help="按概率随机采样；默认贪心 argmax")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=3)
args = parser.parse_args()

if not args.prompt:
    parser.error("prompt 不能为空")
if args.tokens < 0 or args.temperature <= 0 or args.top_k < 1:
    parser.error("tokens 必须 >= 0，temperature > 0，top-k >= 1")

file = Path(__file__).resolve().parent / "checkpoints" / "mini_gpt.pt"
if not file.exists():
    raise FileNotFoundError(f"找不到 {file}，请先运行 09_train.py")

device = "cuda" if torch.cuda.is_available() else "cpu"
# 用 CPU 加载权重，再把模型整体移动到所需设备。
checkpoint = torch.load(file, map_location="cpu", weights_only=True)
config = checkpoint["config"]
vocab = checkpoint["vocab"]
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

unknown = sorted(set(args.prompt) - set(stoi))
if unknown:
    raise ValueError(f"提示词有训练词表外的字符: {unknown}；可用字符: {vocab[1:]}")

model = MiniGPT(**config)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

ids = torch.tensor([[stoi[ch] for ch in args.prompt]], dtype=torch.long, device=device)

with torch.no_grad():
    for _ in range(args.tokens):
        # 长度超过上下文窗口时，只取最后 context_length 个 Token。
        context = ids[:, -config["context_length"]:]
        logits = model(context)[:, -1, :]  # 只读取最后一个位置的预测
        logits[:, stoi["<unk>"]] = float("-inf")

        if args.sample:
            logits = logits / args.temperature
            k = min(args.top_k, len(vocab) - 1)
            top_values, top_indices = torch.topk(logits, k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, top_indices, top_values)
            probs = torch.softmax(filtered, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        ids = torch.cat([ids, next_id], dim=1)

result = "".join(itos[i] for i in ids[0].tolist())
print("设备:", device)
print("加载训练步数:", checkpoint["steps"])
print("输入提示词:", args.prompt)
print("生成结果:", result)
