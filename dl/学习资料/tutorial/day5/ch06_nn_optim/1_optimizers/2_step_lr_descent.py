import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义损失函数：传入参数向量（二元）
def f(x):
    return 0.05 * x[0]**2 + x[1] ** 2

# 定义函数，实现梯度下降法流程，传入 初始参数、优化器、训练迭代次数
def gradient_descent(x, optimizer, lr_scheduler, n_iters):
    # 用列表保存参数点坐标
    x0_list = []
    x1_list = []
    # 用列表保存学习率变化
    lr_list = []
    for i in range(n_iters):
        x0_list.append(x[0].item())
        x1_list.append(x[1].item())
        lr_list.append(optimizer.param_groups[0]['lr'])
        # 1. 前向传播、计算损失
        loss = f(x)
        # 2. 反向传播
        loss.backward()
        # 3. 更新参数
        optimizer.step()
        # 4. 梯度清零
        optimizer.zero_grad()
        # 5. 更新学习率
        lr_scheduler.step()

    return x0_list, x1_list, lr_list

# 定义训练超参数
lr = 0.9
n_iters = 500

# 定义初始参数
x = torch.tensor([-7.0, 2], requires_grad=True)

# 定义优化器和学习率调度器
optimizer = SGD([x], lr=lr)
lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.7)

x0_list, x1_list, lr_list = gradient_descent(x, optimizer, lr_scheduler, n_iters)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(x0_list, x1_list, color='r')

# 等高线
x0, x1 = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
loss = f([x0, x1])
ax[0].contour(x0, x1, loss, colors='gray', levels=30)
ax[0].set_title("梯度下降过程")

ax[1].plot(lr_list, color='k')
ax[1].set_title("学习率衰减过程")

plt.show()