import torch
from torch import nn

from common.load_data import get_digit_data

# 1. 加载数据
x_train, x_test, y_train, y_test = get_digit_data()

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(784, 50),
    nn.ReLU(),
    nn.Linear(50, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Softmax(dim=-1)
)

# 3. 加载模型参数
state_dict = torch.load("../data/nn_example.pt")
model.load_state_dict(state_dict)

# 4. 预测
y_pred_proba = model(x_test)
print(y_pred_proba.shape)

y_pred_label = torch.argmax(y_pred_proba, dim=-1)
print(y_pred_label.shape)

# 5. 模型评估：计算准确率
acc_cnt = (y_pred_label == y_test).sum().item()
print("预测准确率: ", acc_cnt / len(y_test))