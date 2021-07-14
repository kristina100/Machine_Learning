import numpy as np
import pandas as pd
from LinearRegression import SelfLinearRegression
import Evaluation_Metrics as em
# 导包获取糖尿病数据集
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

data_diabetes = load_diabetes()

data = data_diabetes['data']
target = data_diabetes['target']
feature_names = data_diabetes['feature_names']

# 三个数据都是numpy的一维数据形式，组合成dataframe，更直观地观察数据
df = pd.DataFrame(data, columns=feature_names)

# 使用sklearn分割数据集
train_X, test_X, train_Y, test_Y = train_test_split(data, target, train_size=0.8)

# 使用梯度下降来测试
lr1 = SelfLinearRegression(solve='sgd')
lr1.fit(train_X, train_Y)
predict = lr1.predict(test_X)
predict1 = predict[0]
# 查看w
print('w', lr1.get_params())
mSE = em.MSE(test_Y, predict1)
rMSE = em.RMSE(test_Y, predict1)
mAE = em.MAE(test_Y, predict1)
rSQUARE = em.RSquare(test_Y, predict1)
print(f"\n均方误差：{mSE}\n")
print(f"均方根误差：{rMSE}\n")
print(f"平均绝对误差：{mAE}\n")
print(f"平方误差：{rSQUARE}\n")

# 使用简单闭包运算
lr2 = SelfLinearRegression(solve='closed')
lr2.fit(train_X, train_Y)
predict2 = lr2.predict(test_X)
mSE = em.MSE(test_Y, predict2)
rMSE = em.RMSE(test_Y, predict2)
mAE = em.MAE(test_Y, predict2)
rSQUARE = em.RSquare(test_Y, predict2)
print(f"\n均方误差：{mSE}\n")
print(f"均方根误差：{rMSE}\n")
print(f"平均绝对误差：{mAE}\n")
print(f"平方误差：{rSQUARE}\n")