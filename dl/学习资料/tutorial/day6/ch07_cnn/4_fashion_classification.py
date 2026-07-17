import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import get_fashion_data

# 1. 加载数据
x_train, x_val, y_train, y_val = get_fashion_data()

# # 显示图片
# img = x_train[12345, 0]
# plt.imshow(img, cmap='gray')
# plt.show()
#
# print(y_train[12345])

# 2. 定义训练超参数
lr = 0.01
epochs = 20
batch_size = 256

# 3. 创建数据加载器
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 4. 搭建模型
model = nn.Sequential(
    nn.Conv2d(1, 6, 5, 1, 2),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(6, 16, 5),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),

    nn.Flatten(),

    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    nn.Sigmoid(),

    nn.Linear(84, 10),
)

# 5. 模型参数初始化
def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
model.apply(init_weights)

# 6. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 7. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 8. 训练和验证
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    train_acc_num = 0

    # 遍历训练集，分批次迭代
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 累加损失及准确个数
        train_loss += loss.item()
        pred_label = output.argmax(dim=-1)
        train_acc_num += pred_label.eq(target).sum().item()
    # 训练完成，计算平均损失和准确率
    this_train_loss = train_loss / len(train_loader)
    this_train_acc = train_acc_num / len(train_dataset)

    # 验证
    model.eval()
    val_acc_num = 0

    # 遍历验证集，分批次迭代
    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)

            # 累加准确个数
            pred_label = output.argmax(dim=-1)
            val_acc_num += pred_label.eq(target).sum().item()
    # 训练完成，计算平均损失和准确率
    this_val_acc = val_acc_num / len(val_dataset)

    print(f'Epoch: {epoch+1} train_loss: {this_train_loss:.6f} train_acc: {this_train_acc:.6f} val_acc: {this_val_acc:.6f}')

# 选取一张图片进行测试
img = x_val[666, 0]
plt.imshow(img, cmap='gray')
plt.show()
print("真实标签：", y_val[666])

x_new = img.unsqueeze(0).unsqueeze(0).to(device)
pred = model(x_new)
pred_label = pred.argmax(dim=-1)
print("预测标签：", pred_label)