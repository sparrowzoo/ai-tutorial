import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import get_digit_data

# 1. 加载数据
x_train, x_val, y_train, y_val = get_digit_data()

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(784, 50),
    nn.ReLU(),
    nn.Linear(50, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    # nn.Softmax(dim=-1)
)
#
# # 参数初始化函数
# def init_params(layer):
#     if isinstance(layer, nn.Linear):
#         # nn.init.xavier_normal_(layer.weight)
#         nn.init.zeros_(layer.weight)
#         nn.init.zeros_(layer.bias)
# model.apply(init_params)

# 3. 定义训练超参数
batch_size = 64
lr = 0.1
epochs = 20

# 4. 构建数据集和加载器
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
# 转成dataloader
# 训练集打散：为了让模型看到不同顺序的样本，防止模型“死记硬背”样本顺序，学习到顺序中的噪声。
# 验证集不打散：为了让评价指标不受顺序干扰，真实反映模型性能，便于追踪训练进展。
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 5. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 6. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device) 将模型搬运到GPU（或CPU），利用GPU的并行计算能力大幅加速训练，同时确保模型参数与数据在相同设备上，避免因设备不一致而报错。
model.to(device)

# 7. 模型训练和验证
for epoch in range(epochs):
    # 7.1 训练
    # model.train()和model.eval()是
    # PyTorch中控制模型行为的“开关”，一句话总结就是：
    # model.train()
    # 告诉模型“现在在训练，请启用Dropout和BatchNorm的动态调整”；
    # model.eval()
    # 告诉模型“现在在验证 / 测试，请固定这些层，用学习到的全局统计量进行推理”。
    model.train()
    train_total_loss = 0
    train_acc_cnt = 0
    for input, target in train_loader:
        #model.to(device)为什么这里还做一次？
        #上边搬的是模型权重（参数），这里搬的是数据
        input, target = input.to(device), target.to(device)
        # 7.1.1 前向传播，计算模型预测输出值
        #因为PyTorch的Model类实现了__call__()这个特殊方法，所以实例对象可以像函数一样被调用。
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