
import torch

torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================
# 1. 准备文本和词表
# ====================================

text = "我喜欢学习。我喜欢数学。"

# 按字符构建词表
# <unk> 表示词表里不存在的字符
vocab = ["<unk>"] + sorted(set(text))

# 字符 → ID
stoi = {ch: i for i, ch in enumerate(vocab)}

# ID → 字符
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi.get(ch, stoi["<unk>"]) for ch in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

print("=== 1. Tokenizer ===")
print("原始文本:", text)
print("词表:", stoi)

ids = encode(text)

print("Token IDs:", ids)
print("解码结果:", decode(ids))

# ====================================
# 2. 模型配置
# ====================================

vocab_size = len(vocab)
d_model = 4
context_length = 16

# (B, T) = (1, 12)
token_ids = torch.tensor([ids], device=device)

B, T = token_ids.shape

assert T <= context_length

# ====================================
# 3. 定义可学习 Embedding 表
# ====================================

# 每个 Token ID 对应一行四维向量
token_table = torch.nn.Parameter(
    torch.randn(
        vocab_size, d_model, device=device
    ) * 0.1
)

# 每个序列位置对应一行四维向量
position_table = torch.nn.Parameter(
    torch.randn(
        context_length, d_model, device=device
    ) * 0.1
)

# ====================================
# 4. 执行 Embedding 查表
# ====================================

token_vectors = token_table[token_ids]

position_ids = torch.arange(T, device=device)

position_vectors = (
    position_table[position_ids].unsqueeze(0)
)

# 内容向量 + 位置向量
X = token_vectors + position_vectors

print("\n=== 2. Embedding Shapes ===")
print("Token IDs:", token_ids.shape)
print("Token Table:", token_table.shape)
print("Token Vectors:", token_vectors.shape)
print("Position Vectors:", position_vectors.shape)
print("最终输入 X:", X.shape)

# ====================================
# 5. 白盒观察同一个 Token
# ====================================

# 文本中位置 0 和位置 6 都是「我」
print("\n=== 3. 同一个 Token，不同位置 ===")

print("位置 0 的 Token ID:", token_ids[0, 0].item())
print("位置 6 的 Token ID:", token_ids[0, 6].item())

print("\n位置 0 的 Token Embedding:")
print(token_vectors[0, 0])

print("\n位置 6 的 Token Embedding:")
print(token_vectors[0, 6])

print("\n位置 0 的最终 X:")
print(X[0, 0])

print("\n位置 6 的最终 X:")
print(X[0, 6])

# ====================================
# 6. 自动验证
# ====================================

assert decode(encode(text)) == text

assert token_ids[0, 0] == token_ids[0, 6]

assert torch.allclose(
    token_vectors[0, 0],
    token_vectors[0, 6]
)

assert torch.allclose(
    X[0, 6] - X[0, 0],
    position_vectors[0, 6] - position_vectors[0, 0]
)

assert X.shape == (1, 12, 4)

print("\nTEST PASSED")


print("\n=== 只看第一个字：我 ===")

print("原始字符:", text[0])
print("Token ID:", token_ids[0, 0].item())
print("Token Vector:", token_vectors[0, 0])
print("Position Vector:", position_vectors[0, 0])
print("最终 X:", X[0, 0])