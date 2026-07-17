import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer   # 列转换器
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

# 加载房屋数据
def get_house_data():
    # 1. 加载数据
    df = pd.read_csv('../data/house_prices.csv')
    df.drop(["Id"], axis=1, inplace=True)

    # 2. 划分数据集
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 特征工程：数值型和类别型分别处理
    num_features = X.select_dtypes(exclude='object').columns
    cat_features = X.select_dtypes(include='object').columns

    # 定义转换器
    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    ct = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
    ])
    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)

    # 4. 转换成Tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train.values).float()
    y_test = torch.tensor(y_test.values).float()

    return x_train, x_test, y_train, y_test


if  __name__ == '__main__':
    # x_train, x_test, y_train, y_test = get_digit_data()
    x_train, x_test, y_train, y_test = get_house_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)