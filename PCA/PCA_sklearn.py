# -*- coding: utf-8 -*- 
# Time : 2021/8/4 15:59 
# Author : Kristina 
# File : PCA_sklearn.py
# contact: kristinaNFQ@163.com
# MyBlog: kristina100.github.io
# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
# 加载鸢尾花数据
data = load_iris()
y = data.target
x = data.data
# 设置要降的维度为2
pca = PCA(n_components=2)
# 用x来训练PCA模型，同时返回降维后的数据
reduced_X = pca.fit_transform(x)
red_x,red_y = [],[]
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i]==1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[1][1])
plt.scatter(red_x, red_y, c='r',marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g',marker='.')
plt.show()