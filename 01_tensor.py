
import torch

# 1. 选择计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 创建输入 Tensor
# B=1, T=3, D=2
x = torch.tensor([
    [
        [1.0, 0.0],  # Token 0
        [0.0, 1.0],  # Token 1
        [1.0, 1.0],  # Token 2
    ]
], device=device)

print("=== 1. 输入 ===")
print("X =", x)
print("Shape =", x.shape)

# 3. 交换最后两个维度
xt = x.transpose(-2, -1)

print("\n=== 2. 转置 ===")
print("X^T =", xt)
print("Shape =", xt.shape)

# 4. 矩阵乘法
scores = x @ xt

print("\n=== 3. Token 关系分数 ===")
print("Scores =", scores)
print("Shape =", scores.shape)

# 5. 取出第二个 Token 的所有关联分数
print("\n=== 4. 观察 Token 1 ===")
print(scores[0, 1, :])

# 6. 验证结果
expected = torch.tensor([[
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 2.],
]], device=device)

assert torch.equal(scores, expected)

print("\nTEST PASSED")


# 7. 把关联分数转为注意力权重
weights = torch.softmax(scores, dim=-1)

print("\n=== 5. 注意力权重 ===")
print(weights)

# 8. 用权重对原始向量进行加权汇总
output = weights @ x

print("\n=== 6. 汇总后的向量 ===")
print(output)
print("Shape =", output.shape)
