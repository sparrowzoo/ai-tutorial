import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import get_house_data

# 1. 加载数据
x_train, x_val, y_train, y_val = get_house_data()

# 2. 定义训练超参数
batch_size = 64
lr = 0.1
epochs = 500

# 3. 构建数据集和加载器
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 4. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 5. 定义模型
feature_num = x_train.shape[1]
model = nn.Sequential(
    nn.Linear(feature_num, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),

    nn.Dropout(0.2),

    nn.Linear(128, 1),
).to(device)

# 参数初始化函数
def init_params(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
model.apply(init_params)

# 6. 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

def loss_fn(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1, float('inf'))
    mse = nn.MSELoss()
    loss = torch.sqrt(mse( torch.log(y_pred), torch.log(y_true)))
    return loss

# 7. 模型训练和验证
train_loss_list = []
val_loss_list = []

for epoch in range(epochs):
    # 7.1 训练
    model.train()
    train_total_loss = 0

    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        # 7.1.1 前向传播，计算模型预测输出值
        output = model(input)
        # 7.1.2 计算损失
        loss = loss_fn(output.squeeze(), target)
        # 7.1.3 反向传播，计算梯度
        loss.backward()
        # 7.1.4 更新参数
        optimizer.step()
        # 7.1.5 梯度清零
        optimizer.zero_grad()
        # 累加损失
        train_total_loss += loss.item()
    # 计算平均训练损失
    train_loss = train_total_loss / len(train_loader)
    train_loss_list.append(train_loss)

    # 7.2 验证
    model.eval()
    val_total_loss = 0

    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            # 7.2.1 前向传播，计算模型预测输出值
            output = model(input)
            # 7.2.2 计算损失
            loss = loss_fn(output.squeeze(), target)

            # 累加损失
            val_total_loss += loss.item()

        # 计算平均训练损失和准确率
        val_loss = val_total_loss / len(val_loader)
        val_loss_list.append(val_loss)

    # 本轮训练和验证结束，打印相关指标
    print(f"Epoch: {epoch+1}   train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")

plt.plot(train_loss_list, "r--", label="train loss", linewidth=3)
plt.plot(val_loss_list, "b-", label="val loss", linewidth=2)
plt.legend()
plt.show()