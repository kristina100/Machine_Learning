# -*- coding: utf-8 -*- 
# Time : 2021/8/3 21:03 
# Author : Kristina 
# File : Application.py
# contact: kristinaNFQ@163.com
# MyBlog: kristina100.github.io
# -*- coding:UTF-8 -*-

import CART_RF as CART
import pprint
filename = 'bcancer.csv'
dataset = CART.load_csv(filename)
# 转换字符型到整型
for i in range(0, len(dataset[0])):
    CART.str_column_to_float(dataset, i)

# 去除索引
dataset_new = []
for row in dataset:
    dataset_new.append([row[i] for i in range(1,len(row))])

# 划分数据集和测试集
training,testing = CART.getTrainTestData(dataset_new, 0.7)
tree = CART.build_tree(training,11,5)
pprint.pprint(tree)

pre = []
act = []
for row in training:
    prediction = CART.predict(tree, row)
    pre.append(prediction)
    actual = act.append(row[-1])

acc = CART.accuracy_metric(act, pre)

print('training accuracy: %.2f'%acc)

for row in testing:
    prediction = CART.predict(tree, row)
    pre.append(prediction)
    actual = act.append(row[-1])
    acc = CART.accuracy_metric(act, pre)

print('testing accuracy: %.2f'%acc)