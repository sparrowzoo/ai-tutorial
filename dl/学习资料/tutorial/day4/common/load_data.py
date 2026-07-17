import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 加载手写数字数据
def get_digit_data():
    # 1. 加载数据
    df = pd.read_csv('../data/train.csv')

    # 2. 划分数据集
    X = df.drop('label', axis=1)
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 特征工程：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 4. 转换成Tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)

    return x_train, x_test, y_train, y_test

if  __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_digit_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)