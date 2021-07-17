from math import log
import operator
from collections import Counter

# 预剪枝
pre_pruning = True
# 后剪枝
post_pruning = True


def read_data():
    fr = open('lenses.txt', 'r')
    all_lines = fr.readlines()
    # 把标签提取出来
    labels = all_lines[0].split()

    # 分离训练集18和测试集6
    trainSet = []
    testSet = []

    for line1 in all_lines[1:-6]:
        line1 = line1.strip().split(' ')
        trainSet.append(line1)

    for line2 in all_lines[-5:]:
        line2 = line2.strip().split(' ')
        testSet.append(line2)

    return trainSet, testSet, labels


# 计算信息熵
def entropy(trainSet):
    numEntries = len(trainSet)
    labelCounts = {}
    # 给所有可能的分类创建字典
    for featVec in trainSet:
        # 每一行最后一个作为标签
        currentLabel = featVec[-1]
        # 如果label不在字典中，则创建键，并赋值为0，否则加1
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 所有可能的分类 labelCounts : {'3': 11, '2': 4, '1': 3}
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)
    # Ent：1.3472230399326601
    return Ent


# 划分数据集
def splitTrainSet(trainSet, axis, value):
    """
        按照给定特征axis，划分数据集
        :param trainSet: 待划分的数据集
        :param axis: 划分数据集的特征(第几个特征)
        :param value: 需要返回的特征的值(相应特征的值)
        :return: 给定特征axis的值(除了相应特征的值没有，剩下的全部都有)
    """
    '''在函数内部对列表对象的修改，将会影响该列表对象的整个生存周期
       所以新建一个列表对象'''
    # 创建返回的数据集列表
    retrainSet = []
    # 遍历数据集
    for featVec in trainSet:
        # 抽取符合划分特征的值
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # 列表末尾一次性追加另一个序列中的多个值
            reducedFeatVec.extend(featVec[axis + 1:])
            # 将符合条件的特征以列表的形式添加到返回的数据集列表
            retrainSet.append(reducedFeatVec)

    return retrainSet


# 选择最好的数据集划分方式
def choose_best_feature_split(trainSet):
    """
       选择最好的数据集划分方式
       :param trainSet: 给定的数据集
       :return: 返回最好的特征
    """
    '''
       划分数据集最大的原则是：将无序的数据变得更加有序
       在划分数据集之前之后信息发生的变化称为信息增益
       获得信息增益最高的特征就是最好的选择
       
       遍历整个数据集，对每个特征划分数据集的结果计算一次信息熵
       通过对比信息熵的大小，找到最好的划分方式
       熵越小说明划分后的数据集越有序
    '''
    # 数据最后一列是类别，不是特征
    num_features = len(trainSet[0]) - 1
    # 计算整个数据集的原始信息熵
    # 保存最初的无序度量值,用于划分完之后的数据集计算的熵值进行比较
    baseEnt = entropy(trainSet)
    # 初始化信息增益和特征
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历数据集中的所有特征
    for i in range(num_features):
        # 将数据集中所有第i个特征值写入新的列表中
        featList = [demo[i] for demo in trainSet]
        # 剔除重复值
        uniqueVals = set(featList)
        newEnt = 0.0
        # 遍历当前特征中所有唯一属性值
        for value in uniqueVals:
            # 对每一个特征划分一次数据集
            singleTrainSet = splitTrainSet(trainSet, i, value)
            # 计算特征值value对应子集占数据集的比例
            p = len(singleTrainSet) / float(len(trainSet))
            # 对所有唯一特征值得到的熵求和
            newEnt += p * entropy(singleTrainSet)

        infoGain = baseEnt - newEnt
        # 比较所有特征中信息增益，并返回最小熵对应的特征划分的索引值
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


# C4.5算法
def C45_chooseBestFeatureToSplit(trainSet):
    numFeatures = len(trainSet[0]) - 1
    baseEnt = entropy(trainSet)
    bestInfoGain_ratio = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        featList = [example[i] for example in trainSet]
        uniqueVals = set(featList)  # 将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt = 0.0
        IV = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subdataset = splitTrainSet(trainSet, i, value)
            p = len(subdataset) / float(len(trainSet))
            newEnt += p * entropy(subdataset)
            IV = IV - p * log(p, 2)
        infoGain = baseEnt - newEnt
        if (IV == 0):  # fix the overflow bug
            continue
        infoGain_ratio = infoGain / IV  # 这个feature的infoGain_ratio
        print(u"C4.5中第%d个特征的信息增益率为：%.3f" % (i, infoGain_ratio))
        if (infoGain_ratio > bestInfoGain_ratio):  # 选择最大的gain ratio
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i  # 选择最大的gain ratio对应的feature
    return bestFeature


def majorityCnt(classList):
    """
        多数表决器
        :param classList: 输入类标签列表
        :return: 返回出现最多次数的标签名称(key)
    """
    classCount = {}
    # 遍历所有的标签列表
    for label in classCount.keys():
        # 存在就值加一，不存在就新增键
        if label not in classList:
            classCount[label] = 0
        classCount[label] += 1
    # 将字典中的键值从大到小进行排序
    # sortedClassCount = sorted(classCount.items(), key=lambda item:item[1])
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTreeID3(trainSet, labels, testSet):
    # 类别集合
    # ['3', '2', '3', '1', '3', '2', '3', '1', '3', '2', '3', '1', '3', '2', '3', '3', '3', '3']
    classList = []
    for example in trainSet:
        classList.append(example[-1])
    # 类别完全相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 只剩下一个索引的时候，说明结束
    if len(trainSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的标签名称
        return majorityCnt(classList)
    # 划分数据集，得到最好的特征
    bestFeat = choose_best_feature_split(trainSet)
    # 得到最好的标签
    bestFeatLabel = labels[bestFeat]
    print(f"此时最优的索引是：{bestFeatLabel}")
    # 不断分支 存储了树所有的信息
    ID3Tree = {bestFeatLabel: {}}
    # del 删除的是变量 删掉最好的特性的标签
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in trainSet]
    # 得到唯一表示的属性值
    uniqueVals = set(featValues)

    # 预剪枝
    if pre_pruning:
        # 训练集每一条数据类别的集合
        ans = []
        for index in range(len(testSet)):
            # 提取类别
            ans.append(testSet[index][-1])
        # counter作用就是遍历所有元素，将元素出现的次数记下来
        result_counter = Counter()
        for vec in trainSet:
            # 遍历训练集中所有的类别并计数
            result_counter[vec[-1]] += 1
        # 统计出现次数最多的类别 例子：print(temp.most_common(1))   #[(9, 3)]  元素“9”出现3次
        leaf_output = result_counter.most_common(1)[0][0]
        # 划分前
        root_acc = cal_acc(test_output=[leaf_output] * len(testSet), label=ans)
        outputs = []
        ans = []
        # 在训练集筛选出来的属性值中，
        for value in uniqueVals:
            # 划分数据集
            cut_testSet = splitTrainSet(testSet, bestFeat, value)
            cut_trainSet = splitTrainSet(trainSet, bestFeat, value)
            # ans是训练集中所有分类的总集
            for vec in cut_testSet:
                ans.append(vec[-1])
            result_counter = Counter()
            for vec in cut_trainSet:
                # 计数出现类别的次数
                result_counter[vec[-1]] += 1
            # 得到最多的那个类别
            leaf_output = result_counter.most_common(1)[0][0]
            # 得到最多的类别，分别乘以测试集的条数
            outputs += [leaf_output] * len(cut_testSet)
        cut_acc = cal_acc(test_output=outputs, label=ans)


        if cut_acc <= root_acc:
            # 禁止划分
            return leaf_output

    # 否则就继续划分
    for value in uniqueVals:
        # 复制类别标签
        singleLabels = labels[:]
        # 递归调用函数createTree()
        ID3Tree[bestFeatLabel][value] = createTreeID3(splitTrainSet(trainSet, bestFeat, value),
                                                   singleLabels,
                                                   splitTrainSet(testSet, bestFeat, value))

    # if post_pruning:
    #     tree_output = cl
    #

    return ID3Tree

def createtreeC45(trainSet, labels, testSet):
    # 类别集合
    # ['3', '2', '3', '1', '3', '2', '3', '1', '3', '2', '3', '1', '3', '2', '3', '3', '3', '3']
    classList = []
    for example in trainSet:
        classList.append(example[-1])
    # 类别完全相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 只剩下一个索引的时候，说明结束
    if len(trainSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的标签名称
        return majorityCnt(classList)
    # 划分数据集，得到最好的特征
    bestFeat =C45_chooseBestFeatureToSplit(trainSet)
    # 得到最好的标签
    bestFeatLabel = labels[bestFeat]
    print(f"此时最优的索引是：{bestFeatLabel}")
    # 不断分支 存储了树所有的信息
    C45Tree = {bestFeatLabel: {}}
    # del 删除的是变量 删掉最好的特性的标签
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in trainSet]
    # 得到唯一表示的属性值
    uniqueVals = set(featValues)

    # 预剪枝
    if pre_pruning:
        # 训练集每一条数据类别的集合
        ans = []
        for index in range(len(testSet)):
            # 提取类别
            ans.append(testSet[index][-1])
        # counter作用就是遍历所有元素，将元素出现的次数记下来
        result_counter = Counter()
        for vec in trainSet:
            # 遍历训练集中所有的类别并计数
            result_counter[vec[-1]] += 1
        # 统计出现次数最多的类别 例子：print(temp.most_common(1))   #[(9, 3)]  元素“9”出现3次
        leaf_output = result_counter.most_common(1)[0][0]
        # 划分前
        root_acc = cal_acc(test_output=[leaf_output] * len(testSet), label=ans)
        outputs = []
        ans = []
        # 在训练集筛选出来的属性值中，
        for value in uniqueVals:
            # 划分数据集
            cut_testSet = splitTrainSet(testSet, bestFeat, value)
            cut_trainSet = splitTrainSet(trainSet, bestFeat, value)
            # ans是训练集中所有分类的总集
            for vec in cut_testSet:
                ans.append(vec[-1])
            result_counter = Counter()
            for vec in cut_trainSet:
                # 计数出现类别的次数
                result_counter[vec[-1]] += 1
            # 得到最多的那个类别
            leaf_output = result_counter.most_common(1)[0][0]
            # 得到最多的类别，分别乘以测试集的条数
            outputs += [leaf_output] * len(cut_testSet)
        cut_acc = cal_acc(test_output=outputs, label=ans)

        if cut_acc <= root_acc:
            # 禁止划分
            return leaf_output

    # 否则就继续划分
    for value in uniqueVals:
        # 复制类别标签
        singleLabels = labels[:]
        # 递归调用函数createTree()
        C45Tree[bestFeatLabel][value] = createTreeID3(splitTrainSet(trainSet, bestFeat, value),
                                                   singleLabels,
                                                   splitTrainSet(testSet, bestFeat, value))

    # if post_pruning:
    #     tree_output = cl
    #

    return C45Tree

def classifytest(inputTree, featLabels, testTrainSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testTrainSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    # 递归遍历整棵树
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def cal_acc(test_output, label):
    """
    :param test_output: the output of testset
    :param label: the answer
    :return: the acc of
    """
    try:
        assert len(test_output) == len(label)
        count = 0.0
        for index in range(len(test_output)):
            if test_output[index] == label[index]:
                count += 1

        return float(count / len(test_output))
    except Exception as a:
        print(a)


if __name__ == '__main__':
    trainSet, testSet, labels = read_data()
    print(f"trainSet{trainSet}")
    print("---------------------------------------------")
    print(f"数据集长度{len(trainSet)}")
    print(f"Ent(D):{entropy(trainSet)}")
    print("---------------------------------------------")
    print(f"下面开始创建的决策树-------")


    # 拷贝，createTree会改变labels
    labels_tmp1 = labels[:]
    ID3tree = createTreeID3(trainSet, labels, testSet)
    print(f"ID3tree_TrainSet:\n{ID3tree}")
    print("---------------------------------------------")

    labels_tmp2 = labels[:]  # 拷贝，createTree会改变labels
    C45desicionTree = createtreeC45(trainSet, labels_tmp2, testSet)
    print('C45desicionTree:\n', C45desicionTree)
