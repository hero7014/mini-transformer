
import math
import torch

torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================
# 1. 定义输入
# ====================================

X = torch.tensor([[
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 1., 1.]
]], device=device)

B, T, D = X.shape

num_heads = 2
head_dim = D // num_heads

assert D % num_heads == 0

# ====================================
# 2. 定义参数矩阵
# ====================================

W_Q = torch.nn.Parameter(
    torch.randn(D, D, device=device) * 0.2
)
W_K = torch.nn.Parameter(
    torch.randn(D, D, device=device) * 0.2
)
W_V = torch.nn.Parameter(
    torch.randn(D, D, device=device) * 0.2
)
W_O = torch.nn.Parameter(
    torch.randn(D, D, device=device) * 0.2
)

# ====================================
# 3. Multi-Head Attention
# ====================================

def multi_head_attention(x):
    B, T, D = x.shape

    # 生成 Q、K、V
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    # 将特征维度拆分成多个 Head
    def split_heads(tensor):
        tensor = tensor.reshape(
            B, T, num_heads, head_dim
        )
        return tensor.transpose(1, 2)

    Q = split_heads(Q)
    K = split_heads(K)
    V = split_heads(V)

    # Q、K、V shape:
    # (B, num_heads, T, head_dim)

    # 各个 Head 独立计算分数
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    # Causal Mask
    mask = torch.triu(
        torch.ones(
            T, T,
            device=x.device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    scores = scores.masked_fill(
        mask, float("-inf")
    )

    # Softmax
    weights = torch.softmax(scores, dim=-1)

    # 每个 Head 汇总 V
    head_output = weights @ V

    # (B, H, T, d_head)
    # 转换为 (B, T, H, d_head)
    merged = head_output.transpose(1, 2)

    # Concat：合并 Head 维度和特征维度
    concat = merged.contiguous().reshape(B, T, D)

    # 输出投影
    output = concat @ W_O

    return weights, head_output, concat, output


# ====================================
# 4. 执行和观察
# ====================================

weights, heads, concat, output = multi_head_attention(X)

print("输入 X:", X.shape)
print("注意力权重:", weights.shape)
print("各 Head 输出:", heads.shape)
print("Concat:", concat.shape)
print("最终输出:", output.shape)

print("\n=== Head 1 Weights ===")
print(weights[0, 0])

print("\n=== Head 2 Weights ===")
print(weights[0, 1])

print("\n=== Concat ===")
print(concat)

print("\n=== Final Output ===")
print(output)


# ====================================
# 5. 自动验证
# ====================================

assert weights.shape == (1, 2, 3, 3)
assert heads.shape == (1, 2, 3, 2)
assert concat.shape == (1, 3, 4)
assert output.shape == (1, 3, 4)

# 每个 Head 的注意力权重按行求和为 1
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones_like(weights.sum(dim=-1))
)

# 修改未来 Token，验证第一个位置不受影响
X_changed = X.clone()
X_changed[:, 1:, :] += 10

_, _, _, changed_output = multi_head_attention(X_changed)

assert torch.allclose(
    output[:, 0, :],
    changed_output[:, 0, :],
    atol=1e-6
)

print("\nTEST PASSED")
