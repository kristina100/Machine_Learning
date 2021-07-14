import pandas as pd
from LogisticRegression import SelfLogisticRegression
import Evalution_Metrics as em
# Guide package to get Iris dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data_diabetes = load_iris()

data = data_diabetes['data']
target = data_diabetes['target']
feature_names = data_diabetes['feature_names']

# All three data are in the form of numpy one-dimensional data,
# ombined into a dataframe for a more intuitive view of the data
df = pd.DataFrame(data, columns=feature_names)

# Splitting datasets using sklearn
train_X, test_X, train_Y, test_Y = train_test_split(data, target, train_size=0.8)

# Use gradient descent to test
lr1 = SelfLogisticRegression(solver='sgd')
lr1.fit(train_X, train_Y)
predict = lr1.predict(test_X)
predict1 = predict[0]
# View w
print('w', lr1.get_params())
mSE = em.MSE(test_Y, predict1)
rMSE = em.RMSE(test_Y, predict1)
mAE = em.MAE(test_Y, predict1)
rSQUARE = em.RSquare(test_Y, predict1)
print(f"\nMean Square Error:{mSE}\n")
print(f"Root mean square error.{rMSE}\n")
print(f"Mean absolute error:{mAE}\n")
print(f"Square Error.{rSQUARE}\n")

# Using Simple Closure Operations
lr2 = SelfLogisticRegression(solver='closed')
lr2.fit(train_X, train_Y)
predict2 = lr2.predict(test_X)
mSE = em.MSE(test_Y, predict2)
rMSE = em.RMSE(test_Y, predict2)
mAE = em.MAE(test_Y, predict2)
rSQUARE = em.RSquare(test_Y, predict2)
print(f"\nMean Square Error:{mSE}\n")
print(f"Root mean square error:{rMSE}\n")
print(f"Mean absolute error:{mAE}\n")
print(f"Square Error.{rSQUARE}\n")