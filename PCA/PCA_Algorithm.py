# -*- coding: utf-8 -*- 
# Time : 2021/8/4 14:45 
# Author : Kristina 
# File : PCA_Algorithm.py
# contact: kristinaNFQ@163.com
# MyBlog: kristina100.github.io
# -*- coding:UTF-8 -*-

import numpy as np


def zeroMean(dataMat):
    """
    零均值化：就是求每一列的平均值，然后该列上的所有数都减去这个均值
    也就是说，零均值化是对每一个特征而言，零均值化之后每一个特征值的均值是0
    :param dataMat: 原始数据集， 每一行代表一个样本
                    每一列代表同一个特征
    :return:
    """
    # axis=0 表示按照列求均值
    # 8.50000,12.16667,7.83333,10.00000,10.50000,6.66667,15.33333,10.33333
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, percent=0.19):
    """

    :param dataMat:
    :param percent:
    :return:
    """
    newData, meanVal = zeroMean(dataMat)
    '''
    求协方差矩阵
    若rowvar=0说明传入的数据一行代表一个样本
    若rowvar=1说明传入的数据一列代表一个样本
    '''
    covMat = np.cov(newData, rowvar=False)
    '''
    线性代数模块lianlg中的eig函数用来计算特征向量
    np.mat()将列表矩阵化
    eigVals 存放特征值
    eigVects 存放行向量，每一列代表一个特征向量
    特征值和特征向量意义对应
    '''
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    n = percentage2n(eigVals, percent)
    print('\n' + str(n) + u'vectors\n')
    '''
    保留主要成分：即保留值比较大的前n个特征
    前面已经得到了eigVals，假设里面有m个特征值，
    对其进行排序，排在前面的n特征值所对应的特征向量是要保留的
    他们组成一个新的特征空间的一组基n_eigVect
    将零均值化的数据乘以n_eigVect就可以得到降维后的数据。
    '''
    # 对特征值从大到小进行排序
    eigValIndice = np.argsort(eigVals)
    # 最大的n个特征值的下标
    # 前n个，逆序取出[start, stop, step]
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    # 最大的n个特征值对应的特征向量
    n_eigVect = eigVects[:, n_eigValIndice]
    # 低维的特征空间的数据
    lowDDataMat = newData*n_eigVect
    # 重构数据
    reconMat = (lowDDataMat*n_eigVect.T)+meanVal
    return reconMat, lowDDataMat, n


def percentage2n(eigVals, percentage):
    """

    :param eigVals:
    :param percentage:
    :return:
    """
    '''
    通过百分比来确定n
    '''
    # 升序
    sortArray = np.sort(eigVals)
    # 逆转 降序
    sortArray = sortArray[-1::-1]
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


if __name__ =='__main__':
    data = np.random.randint(1, 20, size=(6,8))
    print(data)
    fin = pca(data, 0.9)
    mat = fin[1]
    print(mat)
