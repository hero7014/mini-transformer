"""实验 09：使用下一个字符作为标签训练 Mini GPT。"""
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_gpt_model import MiniGPT


torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "checkpoints" / "mini_gpt.pt"

# 仅供演示的小语料：模型会主要记住这两句话的模式。
text = "我喜欢学习。我喜欢数学。"
corpus = text * 200
vocab = ["<unk>"] + sorted(set(corpus))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in corpus], dtype=torch.long, device=device)

config = dict(
    vocab_size=len(vocab), d_model=32, num_heads=4,
    hidden_dim=128, num_layers=2, context_length=16,
)
model = MiniGPT(**config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

batch_size = 32
max_steps = int(os.getenv("STEPS", "600"))
T = config["context_length"]
offsets = torch.arange(T, device=device)


def get_batch():
    # 随机抽取 32 个长度为 T+1 的连续片段。
    starts = torch.randint(0, data.numel() - T, (batch_size,), device=device)
    x = data[starts[:, None] + offsets[None, :]]
    y = data[starts[:, None] + offsets[None, :] + 1]
    return x, y


print("设备:", device)
print("词表:", stoi)
print("模型参数量:", sum(p.numel() for p in model.parameters()))

x_demo, y_demo = get_batch()
print("\n输入示例:", "".join(itos[i] for i in x_demo[0].tolist()))
print("目标示例:", "".join(itos[i] for i in y_demo[0].tolist()))

model.train()
for step in range(1, max_steps + 1):
    x, targets = get_batch()

    # Forward: 每个位置对全词表输出一个 Logit
    logits = model(x)  # (B, T, V)

    # 训练目标：每个位置的“下一个字符”
    loss = F.cross_entropy(
        logits.reshape(-1, config["vocab_size"]),
        targets.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0 or step == max_steps:
        print(f"Step {step:>3}/{max_steps} | Loss = {loss.item():.4f}")

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "config": config,
    "vocab": vocab,
    "steps": max_steps,
}, CHECKPOINT)
print("\n模型已保存:", CHECKPOINT)
