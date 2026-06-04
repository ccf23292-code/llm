# 手写数字识别
# W要变成（特征值，10）
# label变独热   
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
data = load_digits()
X, y = data.data, data.target

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# X是（m,64），且范围在0~16，首先缩放到0-1
X_train = X_train / 16
X_test = X_test / 16
y_train = np.eye(10)[y_train] # 变成独热码，比如3->[0,0,0,1,0...]
y_test = np.eye(10)[y_test]
W = np.zeros((64,10), dtype = float)
b = np.zeros((1,10), dtype = float)
m = X_train.shape[0]
lr = 0.5
epochs = 5000

def softmax(Z):   # z的shape是（m,10）
    Z = Z - Z.max(axis = 1, keepdims = True)
    return np.exp(Z) / np.sum(np.exp(Z), axis = 1, keepdims = True)
for epoch in range (epochs):
    Z = X_train @ W + b
    y_pred = softmax(Z)
    dW = (1/m) * X_train.T @ (y_pred - y_train)
    db = (1/m) * np.sum(y_pred - y_train, axis = 0, keepdims = True)
    W -= lr * dW
    b -= lr * db
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    loss = -np.mean(np.sum(y_train * np.log(y_pred), axis=1))
    if ((epoch+1) % 500 == 0 ):
        print(f"Epochs: [{epoch+1}/{epochs}] , loss: {loss:.4f}")

pred_train = np.argmax(softmax(X_train @ W + b), axis = 1)
true_train = np.argmax(y_train, axis = 1)
accuracy_train = (pred_train == true_train).mean()
print(f"{accuracy_train = }")

pred_test = np.argmax(softmax(X_test @ W + b), axis = 1)
true_test = np.argmax(y_test, axis = 1)
accuracy_test = (pred_test == true_test).mean()
print(f"{accuracy_test = }")

np.save('digit_recognition_W',W)
np.save('digit_recognition_b',b)