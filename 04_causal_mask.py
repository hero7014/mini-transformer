
import torch
import math

torch.set_printoptions(precision=4, sci_mode=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 输入 X：B=1, T=3, D=2
X = torch.tensor([[
    [1., 0.],
    [0., 1.],
    [1., 1.]
]], device=device)

# 2. 手工指定三个参数矩阵
W_Q = torch.tensor([
    [1., 1.],
    [0., 1.]
], device=device)

W_K = torch.tensor([
    [1., 0.],
    [0., 2.]
], device=device)

W_V = torch.tensor([
    [0., 1.],
    [1., 0.]
], device=device)

# 3. 生成 Q、K、V
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("=== Q ===")
print(Q)

print("\n=== K ===")
print(K)

print("\n=== V ===")
print(V)

# 4. Q 与 K 计算匹配分数
scores = Q @ K.transpose(-2, -1)

# 5. 缩放
d_k = Q.shape[-1]
scores = scores / math.sqrt(d_k)

print("\n=== Scores ===")
print(scores)


# 获取 Token 数量
T = X.shape[1]

# 生成上三角掩码（不包含对角线）
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool, device=device),
    diagonal=1
)

print("\n=== Causal Mask ===")
print(mask)

# 将未来位置的分数设为负无穷
scores = scores.masked_fill(mask, float("-inf"))

print("\n=== Masked Scores ===")
print(scores)

# 再进行 Softmax
weights = torch.softmax(scores, dim=-1)

print("\n=== Masked Attention Weights ===")
print(weights)


# 7. 使用 V 汇总信息
output = weights @ V

print("\n=== Output ===")
print(output)
print("Shape:", output.shape)

# 8. 验证形状及权重
assert output.shape == (1, 3, 2)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones_like(weights.sum(dim=-1))
)

print("\nTEST PASSED")
