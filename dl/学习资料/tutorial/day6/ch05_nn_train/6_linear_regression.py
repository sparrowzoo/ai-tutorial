import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 1. 准备数据
# 1.1 定义x和y
x = torch.randn(100, 1)
noise = torch.randn(100, 1) * 0.1
y = 2.5 * x + 5.2 + noise
# print(x)
# print(y)

# 1.2 构建数据集
dataset = TensorDataset(x, y)

# 1.3 定义数据加载器
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. 构建模型
model = nn.Linear(1, 1)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 模型训练
loss_list = []
# 循环训练多轮
for epoch in range(100):
    total_loss = 0

    # 每个轮次，遍历所有批次进行迭代
    for x_batch, y_batch in dataloader:
        # 4.1 前向传播，计算模型预测输出值
        y_pred = model(x_batch)
        # 4.2 计算损失
        loss = loss_fn(y_pred, y_batch)
        # 4.3 反向传播，计算梯度
        loss.backward()
        # 4.4 更新参数
        optimizer.step()
        # 4.5 梯度清零
        optimizer.zero_grad()
        # 累加损失
        total_loss += loss.item()
    # 将本轮平均损失添加到列表
    loss_list.append( total_loss / len(dataloader) )

print("训练完成时的模型参数：")
print(model.weight)
print(model.bias)

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# 1. 训练误差随轮次下降曲线
ax[0].plot(loss_list)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
# 2. 散点图和拟合直线
ax[1].scatter(x, y)
y_pred = model(x)
ax[1].plot(x, y_pred.detach(), color='red')
plt.show()