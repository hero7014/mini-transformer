
import math
import torch

torch.manual_seed(7)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 输入数据和期望输出
X = torch.tensor([[
    [1., 0.],
    [0., 1.],
    [1., 1.]
]], device=device)

target = torch.tensor([[
    [0., 1.],
    [1., 0.],
    [1., 1.]
]], device=device)

# 2. 定义可学习参数
W_Q = torch.nn.Parameter(
    torch.randn(2, 2, device=device) * 0.2
)
W_K = torch.nn.Parameter(
    torch.randn(2, 2, device=device) * 0.2
)
W_V = torch.nn.Parameter(
    torch.randn(2, 2, device=device) * 0.2
)

# 3. 优化器管理三个参数矩阵
optimizer = torch.optim.Adam(
    [W_Q, W_K, W_V], lr=0.05
)

# 4. 开始训练
for step in range(201):

    # Forward：前向计算
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(Q.shape[-1])

    weights = torch.softmax(scores, dim=-1)
    output = weights @ V

    # Loss：计算预测误差
    loss = torch.mean((output - target) ** 2)

    # Backward：反向传播
    optimizer.zero_grad()
    loss.backward()

    if step == 0:
        print("初始梯度:")
        print("W_Q:", W_Q.grad.norm().item())
        print("W_K:", W_K.grad.norm().item())
        print("W_V:", W_V.grad.norm().item())

    
    # Update：更新参数
    
    # 下面所有代码都在 for 循环内部

    if step == 0:
        old_WV = W_V.detach().clone()

        print("\n更新前 WV:")
        print(old_WV)

        print("\nWV 的完整梯度:")
        print(W_V.grad)

    # 每轮都更新 WQ、WK、WV
    optimizer.step()

    if step == 0:
        print("\n更新后 WV:")
        print(W_V.detach())

        print("\nWV 实际变化量:")
        print(W_V.detach() - old_WV)

    # 打印训练进度
    if step in [0, 50, 100, 200]:
        print(f"Step {step}: Loss = {loss.item():.6f}")


# 已经退出 for 循环

# 5. 观察训练后的结果
with torch.no_grad():
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.transpose(-2, -1)
    weights = torch.softmax(
        scores / math.sqrt(Q.shape[-1]), dim=-1
    )
    output = weights @ V

    print("\n训练后注意力权重:")
    print(weights)

    print("\n目标输出:")
    print(target)

    print("\n实际输出:")
    print(output)
