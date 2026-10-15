
import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================
# 1. 模型配置
# ====================================

D = 4
H = 2
head_dim = D // H
hidden_dim = 8

assert D % H == 0

X = torch.tensor([[
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 0., 1.]
]], device=device)

# 第三个 Token 略微调整，避免所有特征相同

# ====================================
# 2. 定义可学习参数
# ====================================

def weight(rows, cols):
    return torch.nn.Parameter(
        torch.randn(rows, cols, device=device) * 0.2
    )

# Attention 参数
W_Q = weight(D, D)
W_K = weight(D, D)
W_V = weight(D, D)
W_O = weight(D, D)

# FFN 参数
W1 = weight(D, hidden_dim)
W2 = weight(hidden_dim, D)

b1 = torch.nn.Parameter(
    torch.zeros(hidden_dim, device=device)
)
b2 = torch.nn.Parameter(
    torch.zeros(D, device=device)
)

# LayerNorm
ln1 = torch.nn.LayerNorm(D).to(device)
ln2 = torch.nn.LayerNorm(D).to(device)

# ====================================
# 3. Multi-Head Causal Attention
# ====================================

def multi_head_attention(x):
    B, T, D = x.shape

    def project(W):
        return (
            (x @ W)
            .reshape(B, T, H, head_dim)
            .transpose(1, 2)
        )

    Q = project(W_Q)
    K = project(W_K)
    V = project(W_V)

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

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

    weights = torch.softmax(scores, dim=-1)

    heads = weights @ V

    concat = (
        heads.transpose(1, 2)
        .contiguous()
        .reshape(B, T, D)
    )

    output = concat @ W_O

    return output, weights


# ====================================
# 4. Feed-Forward Network
# ====================================

def ffn(x):
    hidden = x @ W1 + b1
    activated = F.gelu(hidden)
    output = activated @ W2 + b2
    return output


# ====================================
# 5. Transformer Block (Pre-LN)
# ====================================

def transformer_block(x):

    # 第一部分：Attention
    norm1 = ln1(x)
    attn_output, weights = multi_head_attention(norm1)

    # 第一次残差连接
    h = x + attn_output

    # 第二部分：FFN
    norm2 = ln2(h)
    ffn_output = ffn(norm2)

    # 第二次残差连接
    y = h + ffn_output

    return y, weights, attn_output, ffn_output


# ====================================
# 6. 运行并观察
# ====================================

Y, weights, attn_out, ffn_out = transformer_block(X)

print("Input:", X.shape)
print("Attention Weights:", weights.shape)
print("Attention Output:", attn_out.shape)
print("FFN Output:", ffn_out.shape)
print("Block Output:", Y.shape)

print("\n=== Final Output ===")
print(Y)

# ====================================
# 7. 验证因果性
# ====================================

X_changed = X.clone()
X_changed[:, 2, :] += 10

Y_changed, _, _, _ = transformer_block(X_changed)

# 修改第三个 Token，不应影响前两个位置
assert torch.allclose(
    Y[:, :2, :],
    Y_changed[:, :2, :],
    atol=1e-6
)

# ====================================
# 8. 验证反向传播
# ====================================

loss = Y.square().mean()
loss.backward()

print("\nW_Q Gradient Norm:", W_Q.grad.norm().item())
print("W1 Gradient Norm:", W1.grad.norm().item())

print("\nTEST PASSED")


# ====================================
# 9. 白盒观察 FFN
# ====================================

with torch.no_grad():

    # Attention 后的残差结果
    h = X + attn_out

    # FFN 的输入
    z = ln2(h)

    # 第一层线性变换：4 → 8
    hidden = z @ W1 + b1

    # GELU 激活：8 → 8
    activated = F.gelu(hidden)

    # 第二层线性变换：8 → 4
    result = activated @ W2 + b2

    print("\n=== FFN 白盒实验 ===")

    print("\nFFN 输入:")
    print(z[0, 0])
    print("Shape:", z.shape)

    print("\n第一层线性变换后:")
    print(hidden[0, 0])
    print("Shape:", hidden.shape)

    print("\nGELU 激活后:")
    print(activated[0, 0])
    print("Shape:", activated.shape)

    print("\n第二层线性变换后:")
    print(result[0, 0])
    print("Shape:", result.shape)

    assert torch.allclose(result, ffn_out)
    print("\nFFN TEST PASSED")
