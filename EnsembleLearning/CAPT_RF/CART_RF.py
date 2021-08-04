# -*- coding: utf-8 -*- 
# Time : 2021/8/3 17:00 
# Author : Kristina 
# File : Bagging.py
# contact: kristinaNFQ@163.com
# MyBlog: kristina100.github.io
# -*- coding:UTF-8 -*-


import sys
from csv import reader
import numpy as np
from random import randrange

# 读取文件
def load_csv(filename):
    """
    导入csv返回list
    :param filename: 文件名
    :return: 读出的数据集
    """
    dataset = list()
    with open(filename, 'r') as file:
        # reader() 对象将逐行遍历
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# 将string类型的dataset按照列转化为float类型
def str_column_to_float(dataset, column):
    """

    :param dataset: 数据集
    :param column: 列
    :return:
    """
    for row in dataset:
        if row[column] == '?':
            row[column] = 0
        else:
            row[column] = float(row[column].strip())


# 将数据集按照有放回的方式随机划分为k组
def cross_validation_split(dataset, n_folds):
    """

    :param dataset: 数据集
    :param n_folds: 分组数
    :return: 按照folds分割后的数据
    """

    dataset_split = list()
    # 先确定分割大小
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        # 实现又放回
        dataset_copy = list(dataset)
        fold = list()
        while len(fold) < fold_size:
            # 随机划分
            index = randrange(len(dataset_copy))
            # 移除列表，并返回该列表
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# 计算预测结果的精确度
def accuracy_metric(actual, predicted):
    """

    :param actual: 真实数据
    :param predicted: 预测得到的数据
    :return: 精确度
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# 运行算法，返回精度集合
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """

    :param dataset: 数据集
    :param algorithm: 用到的分类器，这里是指决策树算法
    :param n_folds: k折
    :param args: 这里指多个不确定参数
    :return: 对划分数据集进行分类的精确度集合
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# 基于某个特征对数据集进行划分
def createSplit(index, value, dataset):
    """

    :param index: 数组下标
    :param value: 断定的特征值
    :param dataset: 数据集
    :return: 返回两个完成后的分组
    """
    # 初始化两个列表来存储 最高和最低
    left, right = list(), list()

    # 循环浏览属性值，来创建子集
    for values in dataset:
        # 使用阈值
        if values[index] <= value:
            left.append(values)
        else:
            right.append(values)
    return left, right


# 划分训练集和测试集
def getTrainTestData(dataset, split):
    """

    :param dataset:  此时已经去掉了第一列
    :param split:
    :return:
    """
    np.random.seed(0)
    training = []
    testing = []
    # # 多维数组，仅对第一维打乱顺序
    np.random.shuffle(dataset)
    # shape:(699, 10)
    shape = np.shape(dataset)
    # trainlength : 489
    """
     np.floor 功能：返回数字的下舍整数，也就是向下取整

     整型分为 有符号整型 和 无符号整型
     其区别在于 无符号整型 可以存放的正数范围 比 有符号整型 大一倍，
     因为 有符号整型  将最高位存储符号 
     而 无符号整型 全部存储数字
     """
    trainlength = np.uint16(np.floor(split * shape[0]))

    for i in range(trainlength):
        training.append(dataset[i])

    for i in range(trainlength, shape[0]):
        testing.append(dataset[i])

    return training, testing


# 计算某组分割数据集的基尼指数
# 分类任务中决策树利用基尼指数进行分类
def gini_index(groups, class_values):
    """

    :param groups: 划分的样本子集
    :param class_values: 标签
    :return: 该分类下的基尼指数
             总体包含的类别越杂乱，GINI指数就越大
    """
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


# 得到新的节点
def getNode(dataset):
    """

    :param dataset: 无第一列的数据集
    :return: 根据基尼系数划分后的索引index、标签value、数据groups
    """
    # class_values = list(set(row[-1] for row in dataset))
    class_values = []
    for row in dataset:
        class_values.append(row[-1])

    # 提取数据集中独特存在的标签
    class_values = np.unique(np.array(class_values))

    # 初始化变量，以存储基尼分值、属性指数和分割组
    b_index = sys.maxsize
    b_value = sys.maxsize
    b_score = sys.maxsize
    b_groups = None

    # 循环运行来访问每个属性和属性值
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = createSplit(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    # 一次执行完后后将新节点放入字典
    node = {'index': b_index, 'value': b_value, 'groups': b_groups}
    return node


# 构建最终的数据集
def terminalNode(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# 创建子树节点或者终止建树过程
def buildTree(node, max_depth, min_size, depth):
    """

    :param node: 当前节点
    :param max_depth: 树的最大深度
    :param min_size: 叶节点最小的数据量，如果低于这个限制，则视为数据量过小
    :param depth: 当前树的深度
    :return:
    """
    # 先获得子集信息
    left, right = node['groups']
    del (node['groups'])
    # 检查左右组的每一个元素
    if not left or not right:
        # 如果群组中没有元素，调用终端节点，终止建树过程
        combined = left + right
        node['left'] = terminalNode(combined)
        node['right'] = terminalNode(combined)
        return
    # 检查是否达到最大深度
    if depth >= max_depth:
        node['left'] = terminalNode(left)
        node['right'] = terminalNode(right)
        return
    # 如果一切正常，则开始为左边的节点建立树状结构
    # 如果节点小于了最小的实例，则停止进一步构建
    if len(left) <= min_size:
        node['left'] = terminalNode(left)
    else:
        # 在树的左侧创建新节点
        node['left'] = getNode(left)
        # 在树下追加节点并增加一个深度
        # 实现递归
        buildTree(node['left'], max_depth, min_size, depth + 1)

    # 右侧节点也同左侧
    if len(right) <= min_size:
        node['right'] = terminalNode(right)

    else:
        node['right'] = getNode(right)
        buildTree(node['right'], max_depth, min_size, depth + 1)


# 建树操作
def build_tree(train, max_depth, min_size):
    """

    :param train: 训练集
    :param max_depth: 树的深度
    :param min_size: 叶结点最小的数据量
    :return: 根节点
    """
    root = getNode(train)
    buildTree(root, max_depth, min_size, 1)
    return root


# 使用单个决策树进行预测
def predict(node, row):
    """

    :param node:
    :param row:
    :return:
    """
    # 获取节点值并检查属性值是否小于或者等于平均值
    if row[node['index']] <= node['value']:
        # If yes enter into left branch and check whether it has another node or the class value.
        if isinstance(node['left'], dict):
            return predict(node['left'], row)  # Recursion
        else:
            # If there is no node in the branch
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# 打印树
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.2f]' % (depth * ' ', (node['index'] + 1), node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % (depth * ' ', node))


def build_tree_RF(train, max_depth, min_size, nfeatures):
    """

    :param train: 训练集
    :param max_depth: 最大深度
    :param min_size: 最小节点数
    :param nfeatures: 特征集
    :return:
    """
    root = getNodeRF(train, nfeatures)
    buildTreeRF(root, max_depth, min_size, 1, nfeatures)
    return root


# 创建子节点或者终止建树
def buildTreeRF(node, max_depth, min_size, depth, nfeatures):
    """

    :param node: 节点
    :param max_depth: 最大深度
    :param min_size: 最小节点树
    :param depth: 树深度
    :param nfeatures: 特征集
    :return:
    """
    # 先获取子节点集信息
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        combined = left + right
        node['left'] = terminalNode(combined)
        node['right'] = terminalNode(combined)
        return
    if depth >= max_depth:
        node['left'] = terminalNode(left)
        node['right'] = terminalNode(right)
        return
    if len(left) <= min_size:
        node['left'] = terminalNode(left)

    else:
        node['left'] = getNodeRF(left, nfeatures)
        # 在树下追加一个节点并增加一个深度
        buildTree(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = terminalNode(right)

    else:
        node['right'] = getNodeRF(right, nfeatures)
        buildTree(node['right'], max_depth, min_size, depth + 1)


def getNodeRF(dataset, n_features):
    class_values = []
    for row in dataset:
        class_values.append(row[-1])

    # 提取数据集中存在的唯一标签
    class_values = np.unique(np.array(class_values))

    # 初始化变量 存储基尼分值 属性 组
    b_index = sys.maxsize
    b_value = sys.maxsize
    b_score = sys.maxsize
    b_group = None

    # 随机选择特征
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)

    for index in features:
        for row in dataset:
            groups = createSplit(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_group = index, row[index], gini, groups

    node = {'index': b_index, 'value': b_value, 'groups': b_group}
    return node


# 从数据集中创建一个随机子样本，并进行替换
def subsample(dataset, ratio):
    """

    :param dataset: 数据集
    :param ratio:
    :return:
    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# 预测bagging
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# 随机森林算法
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree_RF(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)