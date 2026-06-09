# 数字识别深度网路
# 两个隐藏层，64->32->16->10  ReLU+ReLU+softmax
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
np.random.seed(0)
data = load_digits()
X,y = data.data,data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
X_train = X_train / 16;
X_test = X_test / 16;
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]
m = X_train.shape[0]
lr = 0.5
epochs = 20000

# 定义两层隐藏层的W和b
W1 = np.random.rand(64,32) * 0.1
b1 = np.zeros((1,32))
W2 = np.random.rand(32,16) * 0.1
b2 = np.zeros((1,16))
W3 = np.random.rand(16,10) * 0.1
b3 = np.zeros((1,10))

def ReLU(Z):
    return np.maximum(0,Z)
def softmax(Z):
    Z = Z - Z.max(axis = 1, keepdims = True)
    return np.exp(Z)/np.sum(np.exp(Z),axis = 1, keepdims = True)

for epoch in range(epochs):
    Z1 = X_train @ W1 + b1
    A1 = ReLU(Z1)
    Z2 = A1 @ W2 + b2
    A2 = ReLU(Z2)
    Z3 = A2 @ W3 + b3
    y_pred = softmax(Z3)

    d3 = y_pred - y_train
    dW3 = (1/m) * A2.T @ d3
    db3 = (1/m) * np.sum(d3, axis = 0, keepdims = True)

    d2 = (d3 @ W3.T) * (A2 > 0)
    dW2 = (1/m) * A1.T @ d2
    db2 = (1/m) * np.sum(d2, axis = 0, keepdims = True)

    d1 = (d2 @ W2.T) * (A1 > 0)
    dW1 = (1/m) * X_train.T @ d1
    db1 = (1/m) * np.sum(d1, axis = 0, keepdims = True)

    W3 -= dW3 * lr
    W2 -= dW2 * lr
    W1 -= dW1 * lr
    b3 -= db3 * lr
    b2 -= db2 * lr
    b1 -= db1 * lr
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(np.sum(y_train * np.log(y_pred), axis = 1))
    if((epoch + 1) % 2000 == 0):
        print(f"Epochs: [{epoch+1}/{epochs}] , loss: {loss:.4f}")

# 分析准确率
# 训练集
train_A1 = ReLU(X_train @ W1 + b1)
train_A2 = ReLU(train_A1 @ W2 + b2)
pred_train  = np.argmax(softmax(train_A2 @ W3 + b3),axis = 1)
pred_true = np.argmax(y_train,axis = 1)
train_accuracy  = (pred_train == pred_true).mean()
print(f"{train_accuracy = }")

test_A1 = ReLU(X_test @ W1 + b1)
test_A2 = ReLU(test_A1 @ W2 + b2)
pred_test = np.argmax(softmax(test_A2 @ W3 + b3), axis = 1)
test_true = np.argmax(y_test, axis = 1)
test_accuracy = (pred_test == test_true).mean()
print(f"{test_accuracy = }")

