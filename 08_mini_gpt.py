
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 数据与模型配置
# ==========================================

text = "我喜欢学习。我喜欢数学。"

vocab = ["<unk>"] + sorted(set(text))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

token_ids = torch.tensor(
    [[stoi[ch] for ch in text]],
    dtype=torch.long,
    device=device
)

vocab_size = len(vocab)  # 8
d_model = 4
num_heads = 2
hidden_dim = 8
num_layers = 2
context_length = 16


# ==========================================
# 2. Multi-Head Causal Self-Attention
# ==========================================

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 四个可学习投影矩阵
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.2)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.2)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.2)
        self.W_O = nn.Parameter(torch.randn(d_model, d_model) * 0.2)

    def forward(self, x):
        B, T, D = x.shape

        # 投影并拆分 Head
        def project(W):
            return (
                (x @ W)
                .reshape(B, T, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        Q = project(self.W_Q)
        K = project(self.W_K)
        V = project(self.W_V)

        # Attention Scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # Causal Mask
        mask = torch.triu(
            torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))

        # Attention Weights
        weights = torch.softmax(scores, dim=-1)

        # Attention Output
        heads = weights @ V

        # Concat
        concat = (
            heads.transpose(1, 2)
            .contiguous()
            .reshape(B, T, D)
        )

        # 输出投影
        return concat @ self.W_O


# ==========================================
# 3. FFN
# ==========================================

class FeedForward(nn.Module):

    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.W1 = nn.Parameter(
            torch.randn(d_model, hidden_dim) * 0.2
        )
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.W2 = nn.Parameter(
            torch.randn(hidden_dim, d_model) * 0.2
        )
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        hidden = x @ self.W1 + self.b1
        activated = F.gelu(hidden)
        return activated @ self.W2 + self.b2


# ==========================================
# 4. Transformer Block
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(self, x):

        # Attention + Residual
        h = x + self.attn(self.ln1(x))

        # FFN + Residual
        y = h + self.ffn(self.ln2(h))

        return y


# ==========================================
# 5. Mini GPT
# ==========================================

class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        hidden_dim,
        num_layers,
        context_length
    ):
        super().__init__()

        self.context_length = context_length

        # Token Embedding 表
        self.token_table = nn.Parameter(
            torch.randn(vocab_size, d_model) * 0.1
        )

        # Position Embedding 表
        self.position_table = nn.Parameter(
            torch.randn(context_length, d_model) * 0.1
        )

        # 多个独立的 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)

        # LM Head：d_model → vocab_size
        self.W_LM = nn.Parameter(
            torch.randn(d_model, vocab_size) * 0.2
        )

    def forward(self, ids):
        B, T = ids.shape

        if T > self.context_length:
            raise ValueError("输入超过 context_length")

        # 1. Token Embedding
        token_vectors = self.token_table[ids]

        # 2. Position Embedding
        position_ids = torch.arange(T, device=ids.device)
        position_vectors = (
            self.position_table[position_ids].unsqueeze(0)
        )

        # 3. 合成输入 X
        x = token_vectors + position_vectors

        # 记录白盒执行过程
        trace = [
            ("Token IDs", ids),
            ("Token Vectors", token_vectors),
            ("Position Vectors", position_vectors),
            ("Embedding Output X", x),
        ]

        # 4. 依次执行各个 Block
        for i, block in enumerate(self.blocks):
            x = block(x)
            trace.append((f"Block {i+1} Output", x))

        # 5. Final LayerNorm
        h = self.final_norm(x)
        trace.append(("Final LayerNorm", h))

        # 6. LM Head
        logits = h @ self.W_LM
        trace.append(("Logits", logits))

        return logits, trace


# ==========================================
# 6. 创建模型
# ==========================================

model = MiniGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    context_length=context_length
).to(device)

# ==========================================
# 7. 执行前向计算
# ==========================================

model.eval()

with torch.no_grad():
    logits, trace = model(token_ids)

print("=== Mini GPT 前向执行 ===")
print("原始文本:", text)
print("Token IDs:", token_ids.tolist())

for name, tensor in trace:
    print(f"{name}: {tuple(tensor.shape)}")

# ==========================================
# 8. 查看最后位置的预测
# ==========================================

last_logits = logits[0, -1, :]
probabilities = torch.softmax(last_logits, dim=-1)

predicted_id = torch.argmax(probabilities).item()
predicted_token = itos[predicted_id]

print("\n=== 最后一个位置 ===")
print("Logits:", last_logits)
print("Probabilities:", probabilities)
print("预测 Token ID:", predicted_id)
print("预测 Token:", predicted_token)

# ==========================================
# 9. 自动验证
# ==========================================

assert logits.shape == (1, 12, vocab_size)

# 验证不同 Block 不共用同一份参数
assert (
    model.blocks[0].attn.W_Q.data_ptr()
    != model.blocks[1].attn.W_Q.data_ptr()
)

# 修改最后一个 Token，不能影响前面的预测
changed_ids = token_ids.clone()
changed_ids[0, -1] = stoi["我"]

with torch.no_grad():
    changed_logits, _ = model(changed_ids)

assert torch.allclose(
    logits[:, :-1, :],
    changed_logits[:, :-1, :],
    atol=1e-6
)

total_params = sum(p.numel() for p in model.parameters())

print("\nTotal Parameters:", total_params)
print("TEST PASSED")
