import numpy as np
np.random.seed(1)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)   # (4,2)
Y = np.array([[0],   [1],   [1],   [0]],  dtype=float)  # (4,1)  这就是 XOR
m = X.shape[0]

n_hidden = 8
W1 = np.random.randn(2, n_hidden) * 0.5    # ← 形状你来确认：(输入数, 神经元数)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, 1) * 0.5    # ← (隐藏数, 输出数)
b2 = np.zeros((1, 1))

lr = 1.0
epochs = 20000

def sigmoid(z):
    return 1/(1+np.exp(-z))      # 你早会了

for epoch in range(epochs):
    # ===== 前向：4 行（Z1 → A1 → Z2 → y_pred）=====
    #   你写：用 @ 矩阵乘 + 广播加 b + sigmoid
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1) 
    Z2 = A1 @ W2 + b2
    y_pred = sigmoid(Z2)

    # ===== 反向：套"万能配方"，求 6 个量 =====
    #   d2  = ?      输出层误差信号
    d2 = y_pred - Y
    #   dW2, db2 = ?
    dW2 = (1/m) * A1.T @ d2
    db2 = (1/m) * np.sum(d2, axis = 0, keepdims=True)
    #   d1  = ?      误差传回隐藏层（第①跳 @W2.T，第②跳 * 激活导数）
    d1 = (d2 @ W2.T) * (A1 * (1 - A1))
    #   dW1, db1 = ?
    #   你写：照你刚学的配方翻译成代码
    dW1 = (1/m) * X.T @ d1
    db1 = (1/m) * np.sum(d1, axis = 0, keepdims=True)

    # ===== 更新：4 行 =====
    #   你写：每个旋钮 -= lr * 它的梯度
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    loss = -np.mean(Y*np.log(y_pred) + (1-Y)*np.log(1-y_pred))
    if ((epoch+1) % 1000 == 0):
        print(f"Epochs: [{(epoch+1)}/{epochs}], loss: {loss:.4f}")
    # ===== 每 1000 轮打印交叉熵 loss =====
    #   你写
print("预测:", y_pred.round().ravel(), " 真值:", Y.ravel())