import torch
from torch import nn
from torchsummary import summary

if __name__ == '__main__':
    # 定义设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 定义输入数据 (10, 3)
    X = torch.randn(10, 3).to(device)

    # 2. 定义模型
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
        nn.Softmax(dim=-1),
    ).to(device)

    # # 3. 前向传播
    # y = model(X)
    # print("输出预测值：\n", y)

    # 4. 查看参数
    #
    # 4.1 指定层，逐个查看
    # print("\n逐个打印参数：")
    # print("Linear1的参数：")
    # print("权重：", model.linear1.weight)
    # print("偏置：", model.linear1.bias)

    # # 4.2 遍历所有层
    # print("\n遍历打印参数：")
    # for param in model.parameters():
    #     print(param)
    #     print()
    #
    # print()
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param)
    #     print()
    #
    # # 4.3 查看状态字典
    # print("状态字典：")
    # print(model.state_dict())
    #
    # # 5. 查看模型结构和参数数量
    # summary(model, input_size=(3,), batch_size=10, device=device)
    #
    # # 6. 保存模型
    # torch.save(model.state_dict(), 'model.pt')

    # 7. 加载模型参数
    state_dict = torch.load('model.pt')
    model.load_state_dict( state_dict )

    # 8. 使用加载的模型，进行推理预测
    x_new = torch.tensor([1, 2, 3]).float().to(device)
    y_pred = model(x_new)
    print(y_pred)