
import torch

# 1. 查看环境
print("PyTorch:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA GPU 不可用"

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

# 2. 创建两个 GPU 张量
A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
], device=device)

B = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
], device=device)

# 3. 在 GPU 上执行矩阵乘法
C = A @ B

print("\nMatrix Result:")
print(C)
print("Device:", C.device)

# 4. 验证 GPU 计算结果
expected = torch.tensor([
    [19.0, 22.0],
    [43.0, 50.0]
], device=device)

assert torch.allclose(C, expected)
torch.cuda.synchronize()

print("\nGPU TEST PASSED!")
