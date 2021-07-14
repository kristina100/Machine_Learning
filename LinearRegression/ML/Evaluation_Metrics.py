import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """计算准确率"""
    return np.sum(y_true == y_predict) / len(y_true)


def MSE(y_true, y_predict):
    """计算MSE"""
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def RMSE(y_true, y_predict):
    """计算RMSE"""
    return sqrt(MSE(y_true, y_predict))


def MAE(y_true, y_predict):
    """计算MAE"""
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def RSquare(y_true, y_predict):
    """计算R Square"""
    return 1 - MSE(y_true, y_predict) / np.var(y_true)
