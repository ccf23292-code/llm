import numpy as np
np.random.seed(0)
x = np.random.randn(200,1)
true_w,true_b=3.0,-1.0

z_true = true_w * x + true_b

prob = 1 / (1 + np.exp(-1*z_true))

# 这里是生成随机标签Y。np.random.rand(200,1)生成 [0,1) 区间的随机数矩阵，和prob（属于正类的概率）逐元素比较，若随机数<prob则为1，否则为0，相当于根据prob生成一个服从伯努利分布的标签。最后astype(float)转为浮点数类型。
Y = (np.random.rand(200,1) < prob).astype(float)
def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

w,b = 0.0,0.0
lr = 0.1
epochs = 1000
m = len(x)

for epoch in range(epochs):
    z = w * x + b
    y_pred = sigmoid(z)
    dw = 1/m * np.sum((y_pred - Y) * x)
    db = 1/m * np.sum(y_pred - Y)
    w -= dw * lr
    b -= db * lr
    loss = -np.mean((Y*np.log(y_pred)) + (1-Y)*(np.log(1-y_pred)))
    if((epoch+1) % 100 == 0):
        print(f"Epochs: [{(epoch+1)}/{epochs}], loss: {loss:.4f}")

print(f"学出来的 w: {w:.4f}, b: {b:.4f}   (真实 w=3, b=-1)")