import torch
from torch.optim import SGD, RMSprop
import numpy as np
import matplotlib.pyplot as plt

# 定义损失函数：传入参数向量（二元）
def f(x):
    return 0.05 * x[0]**2 + x[1] ** 2

# 定义函数，实现梯度下降法流程，传入 初始参数、优化器、训练迭代次数
def gradient_descent(x, optimizer, n_iters):
    # 用列表保存参数点坐标
    x0_list = []
    x1_list = []
    for i in range(n_iters):
        x0_list.append(x[0].item())
        x1_list.append(x[1].item())
        # 1. 前向传播、计算损失
        loss = f(x)
        # 2. 反向传播
        loss.backward()
        # 3. 更新参数
        optimizer.step()
        # 4. 梯度清零
        optimizer.zero_grad()
    return x0_list, x1_list

# 定义训练超参数
lr = 0.1
n_iters = 500

# 定义初始参数
x = torch.tensor([-7.0, 2], requires_grad=True)

# 对比不同的优化器
# 1. SGD
optimizer = SGD([x], lr=lr)
x0_list, x1_list = gradient_descent(x, optimizer, n_iters)
plt.plot(x0_list, x1_list, color='r')

# 2. RMSProp
# 重新初始化
x = torch.tensor([-7.0, 2], requires_grad=True)

optimizer = RMSprop([x], lr=lr)
x0_list, x1_list = gradient_descent(x, optimizer, n_iters)
plt.plot(x0_list, x1_list, color='b')

# 等高线
x0, x1 = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
loss = f([x0, x1])
plt.contour(x0, x1, loss, colors='gray', levels=30)

plt.legend(['SGD', 'RMSProp'])
plt.show()