import numpy as np
from torch.optim import lr_scheduler

# 1. 造数据（这步不是重点，按你 torch 版翻译过来，直接给你）
np.random.seed(0)
x = np.random.randn(100, 1) * 10
y = 2 * x + 1 + np.random.randn(100, 1) * 2.0

# 2. 初始化
w, b = 0.0, 0.0
lr = 0.01
epochs = 500
m = len(x)

for epoch in range(epochs):
    # a. 前向：算预测 ŷ              ← 你写（就一行：w·x+b）
    y_pred = w * x + b
    # b. 算梯度 dw, db               ← 你写（就是你推的那两个公式）
    dw = 1/m * np.sum((y_pred - y) * x)
    db = 1/m * np.sum(y_pred - y)
    # c. 更新 w, b                   ← 你写（w 减 lr 乘 dw，b 同理）
    w -= dw * lr
    b -= db * lr
    # d. 每 100 轮打印一次 loss
    pass

print(f"学出来的 w: {w:.4f}, b: {b:.4f}")