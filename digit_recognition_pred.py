import numpy as np
from sklearn.datasets import load_digits

# 1. 加载训练好的参数——注意：这里【不训练】！
W = np.load('digit_recognition_W.npy')
b = np.load('digit_recognition_b.npy')

def softmax(Z):                       
    Z = Z - Z.max(axis=1, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

# 2. 拿一张图来认
data = load_digits()
idx = 100
image = data.data[idx]                # (64,) 一张图
true_label = data.target[idx]

# 3. 预处理：必须和训练时一模一样
#  TODO 你写：image 缩放 /16；并 reshape 成 (1,64) 才能 @ W
#            提示：x = (image / 16).reshape(1, -1)
x = (image / 16).reshape(1,-1)
Z = x @ W + b
pred = np.argmax(softmax(Z),axis = 1)
print(f"{pred = }, {true_label = }")