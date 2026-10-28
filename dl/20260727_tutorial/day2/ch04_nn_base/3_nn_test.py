import torch
import torch.nn as nn

# 自定义神经网络类
class MyModel(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        # 定义全连接层
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)
    # 前向传播
    def forward(self, x):
        # 按层顺序依次传递
        x = self.linear1(x)
        x = torch.tanh(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.out(x)
        x = torch.softmax(x, dim=-1)
        return x

if __name__ == '__main__':
    # 1. 定义输入数据 (10, 3)
    X = torch.randn(10, 3)

    # 2. 定义模型
    model = MyModel()

    # 3. 前向传播
    y = model(X)
    print("输出预测值：\n", y)

    # 4. 查看参数

    # 4.1 指定层，逐个查看
    print("\n逐个打印参数：")
    print("Linear1的参数：")
    print("权重：", model.linear1.weight)
    print("偏置：", model.linear1.bias)

    # 4.2 遍历所有层
    print("\n遍历打印参数：")
    for param in model.parameters():
        print(param)
        print()

    print()
    for name, param in model.named_parameters():
        print(name)
        print(param)
        print()

    # 4.3 查看状态字典

    print("状态字典：")
    print(model.state_dict())

    # 5. 查看模型结构和参数数量
    from torchsummary import summary
    summary(model, input_size=(3,), batch_size=10,device="cuda")
    print("修改参数类型",model.linear1.weight.device)

