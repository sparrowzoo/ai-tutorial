import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import get_digit_data

# 1. 加载数据
x_train, x_val, y_train, y_val = get_digit_data()

# 调整形状 (H, C, N, W)
x_train = x_train.reshape(-1, 1, 28, 28)
x_val = x_val.reshape(-1, 1, 28, 28)

# 2. 定义模型
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),

    nn.Flatten(),  # 形状 (N, 32*7*7)

    nn.Linear(32*7*7, 128),
    nn.ReLU(),

    nn.Linear(128, 10),
    # nn.Softmax(dim=-1)
)

# 3. 定义训练超参数
batch_size = 64
lr = 0.1
epochs = 20

# 4. 构建数据集和加载器
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 5. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 6. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 7. 模型训练和验证
for epoch in range(epochs):
    # 7.1 训练
    model.train()
    train_total_loss = 0
    train_acc_cnt = 0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        # 7.1.1 前向传播，计算模型预测输出值
        output = model(input)
        # 7.1.2 计算损失
        loss = loss_fn(output, target)
        # 7.1.3 反向传播，计算梯度
        loss.backward()
        # 7.1.4 更新参数
        optimizer.step()
        # 7.1.5 梯度清零
        optimizer.zero_grad()
        # 累加损失
        train_total_loss += loss.item()
        # 累加预测准确的个数
        pred_label = output.argmax(dim=-1)
        train_acc_cnt += pred_label.eq(target).sum().item()
    # 计算平均训练损失和准确率
    train_loss = train_total_loss / len(train_loader)
    train_acc = train_acc_cnt / len(train_dataset)

    # 7.2 验证
    model.eval()
    val_total_loss = 0
    val_acc_cnt = 0

    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            # 7.2.1 前向传播，计算模型预测输出值
            output = model(input)
            # 7.2.2 计算损失
            loss = loss_fn(output, target)

            # 累加损失
            val_total_loss += loss.item()
            # 累加预测准确的个数
            pred_label = output.argmax(dim=-1)
            val_acc_cnt += pred_label.eq(target).sum().item()

        # 计算平均训练损失和准确率
        val_loss = val_total_loss / len(val_loader)
        val_acc = val_acc_cnt / len(val_dataset)

    # 本轮训练和验证结束，打印相关指标
    print(f"Epoch: {epoch+1}   train loss: {train_loss:.6f}, train acc: {train_acc:.6f}, val loss: {val_loss:.6f}, val acc: {val_acc:.6f}")