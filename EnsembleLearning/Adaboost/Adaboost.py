# -*- coding: utf-8 -*- 
# Time : 2021/8/4 0:34 
# Author : Kristina 
# File : Adaboost.py
# contact: kristinaNFQ@163.com
# MyBlog: kristina100.github.io
# -*- coding:UTF-8 -*-
import sys
import numpy as np


def gini_index(groups, class_values):
    # 初始化基尼指数
    gini = 0.0

    # 计算每一个标签的可能性
    for class_value in class_values:

        # 提取子集
        for group in groups:
            # 组中的实例树
            size = len(group)
            if size == 0:
                continue
                # 初始化一个列表来存储实例的索引
            r = []
            cl = []
            # 获取组中每个实例的类别
            for row in group:
                # 添加权重
                r.append(row[-1])
                # 添加类别
                cl.append(row[-2])

            r = np.array(r)
            # 提取当前行的索引
            class_index = np.where(cl == class_value)

            # 初始化一个变量来增加当前类的权重
            w_add = 0

            # 添加权重
            for w in class_index[0]:
                w_add += r[w]

                # 计算比例
            proportion = w_add / np.sum(r)

            # 计算基尼指数
            gini += (proportion * (1.0 - proportion))
    return gini


def createSplit(attribute, threshold, dataset):
    # 初始化两个列表来存储子集
    lesser, greater = list(), list()

    # 循环浏览属性值，并从中创建子集
    for values in dataset:
        # 使用阈值
        if values[attribute] <= threshold:
            lesser.append(values)
        else:
            greater.append(values)
    return lesser, greater


def getNode(dataset):
    class_values = []
    # 提取存在于数据集中唯一的类值
    for row in dataset:
        # class_values 在倒数第二列
        class_values.append(row[-2])
    class_values = np.unique(class_values)

    # initialize variables to store gini score, attribute index and split groups
    winnerAttribute = sys.maxsize
    attributeValue = sys.maxsize
    gScore = sys.maxsize
    leftGroup = None

    # Run loop to access each attribute and attribute values
    for index in range(len(dataset[0]) - 2):  # leave last two columns
        for row in dataset:
            # Create the groups
            groups = createSplit(index, row[index], dataset)
            # Extract gini score for the threshold
            gini = gini_index(groups, class_values)
            # print('A%d <- %.2f Gini=%.1f' % ((index+1), row[index], gini))
            # If gini score is lower than the previous one choose return it
            if gini < gScore:
                winnerAttribute, attributeValue, gScore, leftGroup = index, row[index], gini, groups

    # Once done create a dictionary for node
    node = {'attribute': winnerAttribute, 'value': attributeValue, 'groups': leftGroup}
    return node


def terminalNode(group):
    outcomes = [row[-2] for row in group]
    return max(set(outcomes), key=outcomes.count)


def decision_stump(dataset):
    # Get node value with best gini score
    node = getNode(dataset)

    # Separate out the groups from the node and remove them
    left, right = node['groups']
    del (node['groups'])

    # Check whether there is any element in the groups or not
    # If there is not any element put the class value with maximum occurence
    if not left or not right:
        node['left'] = node['right'] = terminalNode(left + right)
        return node

    # Put left group's maximum occur class value in left branch
    node['left'] = terminalNode(left)

    # Put right group's maximum occur class value in right branch
    node['right'] = terminalNode(right)
    # print(node)
    return node


def predict(node, row):
    # Get the node value and check whether the attribute value is less than or equal.
    if row[node['attribute']] <= node['value']:
        # If yes enter into left branch and check whether it has another node or the class value.
        # If there is no node in the branch
        return node['left']
    else:
        return node['right']


def getError(actual, predicted, weights):
    # Initialize the error variable
    error = 0

    # We will store the error of each instance in a vector
    error_vec = []

    # Run a loop to calculate error for each instance
    for i in range(len(actual)):
        diff = predicted[i] != actual[i]
        # Weights multiplication to the difference of actual and predicted values
        error += weights[i] * (diff)

        # Append the difference to the error vector
        error_vec.append(diff)

    return error, error_vec


def AdaBoostAlgorithm(dataset, iterations):
    # Initialize the weights of the size of data set
    weights = np.ones(len(dataset), dtype="float32") / len(dataset)
    dataset = np.array(dataset)

    # Add Weights column to the data set(Now last column will be the weights)
    dataset = np.c_[dataset, weights]

    # Create an empty list to store alpha values
    alphas = []

    # Create a list to add weak learners(decision stumps)
    weaks = []

    er = sys.maxsize
    # Lets run the loop for number of iteration(number of classifiers)
    for itr in range(iterations):

        # Create decision tree from the non weighted data-set
        ds = decision_stump(dataset)

        # Create a list to store the predictions of the decision stump
        pred = []

        # Create a list to store actual outputs
        actual = []

        # Let's predict output for each instance in the data set
        for row in dataset:
            actual.append(row[-2])
            pred.append(predict(ds, row))

        # Here we will find out difference between predicted and actual output
        error, error_vec = getError(actual, pred, weights)

        # If error is equal to 0.5 classifier is not able to classify the data set
        if error == 0.0:
            continue
        eps = sys.float_info.epsilon

        # Let's find out the alpha with the help of error
        alpha = (0.5 * np.log((1 - error) / (error + eps)))

        # Create empty vector to store weight updates
        w = np.zeros(len(weights))

        # Update the weights using alpha value
        for i in range(len(error_vec)):

            # For wrong prediction increase the weights
            if error_vec[i] != 0:
                w[i] = weights[i] * np.exp(alpha)

            # For correct prediction decrease the weights
            else:
                w[i] = weights[i] * np.exp(-alpha)

        # Normalize the weights and update previous weight vector
        weights = w / w.sum()

        # Put the updated weights into the data set by over-writing previous weights
        dataset[:, -1] = weights

        # if error<=er:
        print("\nClassifier %i stats:" % itr)
        print(ds)
        print("Error: %.3f and alpha: %.3f" % (error, alpha))
        er = error
        # Append alpha value to the list to used at the time of testing
        alphas.append(alpha)

        # Append the weak learner to the list
        weaks.append(ds)

    return weaks, alphas


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0