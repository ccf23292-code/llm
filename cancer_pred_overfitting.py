from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures # 把 X 升维，使得训练数量和W数量相近，这样会导致过拟合
import numpy as np
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 30, random_state=0)

# PolynomialFeatures 是用来将输入的特征X升维，使得训练数量和W数量相近，这样会导致过拟合
poly = PolynomialFeatures(degree = 2, include_bias = False) # 升维，也就是都变成两次，axis = 1 扩大
X_train = poly.fit_transform(X_train) # fit的目的是确定升维后的特征矩阵的结构，也就是确定特征矩阵的形状，也就是确定特征矩阵的列数
X_test = poly.transform(X_test) # 这里不用fit，因为X_test的特征矩阵的结构已经确定了

mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
# 特征值差距过大，会造成梯度膨胀，出现极端情况log0
# 进行特征缩放
m = X_train.shape[0]
features = X_train.shape[1]
y_train = y_train.reshape(-1,1) # y的shape改成（569,1）
W = np.zeros((features,1),dtype = float)
b = np.zeros((1,1),dtype = float)
lr = 1.0
epochs = 5000
Lambda = 10.0

def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

for epoch in range(epochs):
    z = X_train @ W + b
    y_pred = sigmoid(z)
    dW = (1/m) * X_train.T @ (y_pred - y_train) + Lambda/m * W # 正则化是为了防止过拟合，防止W过大，防止W过小
    db = (1/m) * np.sum(y_pred - y_train,axis = 0, keepdims = True)
    W -= lr * dW
    b -= lr * db
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    loss = -np.mean(y_train*np.log(y_pred) + (1-y_train)*np.log(1-y_pred))
    if ((epoch+1) % 500 == 0):
        print(f"Epochs: [{epoch+1}/{epochs}] , loss: {loss:.4f}")

pred_train = (sigmoid(X_train @ W + b) >= 0.5)
accuracy_train = (pred_train == y_train).mean()
print(f"{accuracy_train = }")



pred_test = (sigmoid(X_test @ W + b) >= 0.5)
y_test = y_test.reshape(-1,1)
accuracy_test = (pred_test == y_test).mean()   
print(f"{accuracy_test = }")