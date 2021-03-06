# -*- coding: utf-8 -*- 
# Time : 2021/8/5 8:36 
# Author : Kristina 
# File : K_means.py
# contact: kristinaNFQ@163.com
# MyBlog: kristina100.github.io
# -*- coding:UTF-8 -*-

import numpy as np


def kmeans(X, k, maxIt):
    """

    :param X:
    :param k:
    :param maxIt:
    :return:
    """
    # 返回行列维度
    numPoints, numDim = X.shape
    # 增加一列作为分类标记
    dataSet = np.zeros((numPoints, numDim + 1))
    # 所有行 除了最后一列
    dataSet[:, :-1] = X
    # 随机选取k行，包含所有列
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
    # 对中心点分类进行初始化
    centroids[:, -1] = range(1, k + 1)
    iterations = 0
    oldCentroids = None
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print(f"iteration:\n{iterations}")
        print(f"dataSet\n {dataSet}")
        print(f"centroids:\n{centroids}")
        # 不能直接使用等号，不然会指向同一个变量
        oldCentroids = np.copy(centroids)
        iterations += 1
        # 根据数据集以及中心点对数据集的点进行归类
        updateLabels(dataSet, centroids)
        # 更新中心点
        centroids = getCentroids(dataSet, k)
    return dataSet


def shouldStop(oldCenttoids, centroids, iterations, maxIt):
    """
    实现函数循环结束的判断
    当循环次数达到最大值的时候，或者中心点不变化的时候就停止
    :param oldCenttoids: 旧的中心点
    :param centroids: 更新后的中心点
    :param iterations: 迭代次数
    :param maxIt: 最大的迭代次数
    :return:
    """
    if iterations > maxIt:
        return True
    return np.array_equal(oldCenttoids, centroids)


def updateLabels(dataSet, centroids):
    """

    :param dataSet: 数据集
    :param centroids: 中心点
    :return:
    """
    # 返回行（点数），列
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        # 对每一行最后一列进行归类
        dataSet[i:-1] = getLabelFromCLoseCentroid(dataSet[i, :-1], centroids)
    numPoints, numDim = dataSet.shape
    return numPoints, numDim


def getLabelFromCLoseCentroid(dataSetRow, centroids):
    """
    对比一行到中心点的距离，返回距离最短的中心点的label
    :param dataSetRow: dataSet[i,:-1]
    :param centroids: 中心点
    :return:
    """
    # 初始化label为中心点的第一点的label
    label = centroids[0, -1]
    # 初始化足校之为当前行到中心点的低一点的距离值
    # np.linalg.norm计算两个向量的距离
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    # 对中心点的每个点开始循环
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
        print("minDist", minDist)
        return label


def getCentroids(dataSet, k):
    """
    更新中心点
    :param dataSet: 数据集包含标签
    :param k: k个分类
    :return: 返回中心点
    """
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        # 找出最后一列类别为i的行集，即求一个类别里面的所有点
        oneCluser = dataSet[dataSet[:, -1] == i, :-1]
        # axis = 0 对行求均值，并赋值
        result[i - 1, :-1] = np.mean(oneCluser, axis=0)
        result[i - 1, -1] = i
        return result


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
# 将点排成矩阵
testX = np.vstack((x1, x2, x3, x4))
print(testX)
result = kmeans(testX, 2, 10)
print(f"final result:{result}")
