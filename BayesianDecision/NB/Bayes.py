"""目标：识别一条短信是否为垃圾信息
   输入：一条文本信息
   输出：二分类的分类结果
   第一步：收集两组带有标签的信息训练集
   第二步：解析训练中所有信息，提取每个词
   统计每一个词出现在正常信息和垃圾信息的频次

"""

# 实现数据预处理、模型训练、预测等功能
import random
import re
import numpy as np


class NavieBayes:
    """
    朴素贝叶斯模型中需要记录的变量：
    不同类型短信数量、相应单词列表、训练集中不重复单词集合等
    它们在类的构造函数中进行初始化，并在模型训练过程中不断更新
    """

    def textParse(self, bigString):
        """
        接收一个大字符串并将其解析为字符串列表
        :param bigString: 字符串
        :return: 字符串列表
        """
        # \w+ 可以匹配数字、字母、下划线
        # 使用\W+ 可以将字符串分隔开
        # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
        listOfTokens = re.split(r'\W+', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]

    def createVocabList(self, dataSet):
        """
        将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
        :param dataSet: 切割后的样本数据集
        :return: vacabSet：返回不重复的词条列表
        """
        # 创建一个空的不重复列表,将list转化成set利用set来处理函数
        vocabSet = set([])
        for demo in dataSet:
            # 取并集
            # vocabSet = vocabSet | set(demo)
            vocabSet = vocabSet.union(set(demo))
        # 返回一个词汇列表
        return list(vocabSet)

    def setOfwords2vec(self, vocabList, inputSet):
        """
        根据vocabList词汇表，将inputSet向量化，向量的每一个元素是1或者0
        :param vocabList: 词汇表列表
        :param inputSet: 每一个词条列表
        :return: 返回文档内容--0 1 向量
        """
        # 先创建一个其中所含元素都为0的向量
        returnVec = [0] * len(vocabList)
        # 遍历每一个单词
        for word in inputSet:
            # 如果单词存在，就置1
            if word in vocabList:
                # index 列表查找
                returnVec[vocabList.index(word)] = 1
            else:
                print(f"the word:{word} is not in my Vocabulary ~ ")
        # 返回 0 1 向量
        return returnVec

    def classifyNB(self, vec2Classify, p0Vec, p1Vec, pClass1):
        """

        :param vec2Classify: 待分类的测试集词汇数组
        :param p0Vec: 非侮辱类条件概率数组
        :param p1Vec: 侮辱类条件概率数组
        :param pClass1: 文档属于侮辱类的概率
        :return:
        """
        p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
        p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
        if p1 > p0:
            # 侮辱类
            return 1
        else:
            # 非侮辱类
            return 0

    def trainNB(self, trainMatrix, trainCategory):
        """
        朴素贝叶斯分类器训练函数
        :param trainMatrix: 向量构成的矩阵
        :param trainCategory: 训练类别标签向量
        :return:
            p0Vect: 非侮辱类的条件概率数组
            p1Vect: 侮辱类的条件概率数组
            pAbusive: 文档属于侮辱类的概率
        """
        # 记录处理邮件的封数
        numTrainDocs = len(trainMatrix)
        # 记录词汇总数 用于构造稀疏矩阵
        numWords = len(trainMatrix[0])
        # 文档属于侮辱类的概率，正好是0-1矩阵嘛
        pAbusive = sum(trainCategory) / float(numTrainDocs)
        # 构建全一矩阵
        p0Num = p1Num = np.ones(numWords)
        # print(p0Num,p1Num)
        # 拉普拉斯平滑 就是对每个类别下所有划分的计数加1,防止从未出现的特征置为0
        p0Denom = p1Denom = 1.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                # 构造非侮辱类词汇矩阵
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                # 构造侮辱类矩阵
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        p1Vect = np.log(p1Num / p1Denom)
        p0Vect = np.log(p0Num / p0Denom)
        return p0Vect, p1Vect, pAbusive

    def spamtest(self):
        docList = []
        classList = []
        fullText = []

        # 遍历25个txt文件
        for i in range(1, 26):
            wordList = self.textParse(open(f"spam/{i}.txt", 'r').read())
            docList.append(wordList)
            fullText.append(wordList)
            # 1 标记非垃圾邮件
            classList.append(1)
            wordList = self.textParse(open(f"ham/{i}.txt", 'r').read())
            docList.append(wordList)
            fullText.append(wordList)
            # 0 标记垃圾邮件
            classList.append(0)
        # 一共有694个词汇
        vocablist = self.createVocabList(docList)
        # 给每封邮件添加索引值  这一步是为了可以一一对应
        trainingSet = list(range(50))
        # 创建存储训练集的索引值的列表和测试集的索引值的列表
        testSet = []
        for i in range(10):
            # 随机挑选10封邮件作为测试集 uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
            randIndex = int(random.uniform(0, len(trainingSet)))
            # randIndex = random.randint(0, len(trainingSet))
            testSet.append(trainingSet[randIndex])
            # 再删除训练集中剪掉的测试集
            del (trainingSet[randIndex])
        # 创建训练集矩阵---稀疏矩阵
        trainMat = []
        # 创建训练集类别标签系向量
        trainClasses = []
        # 遍历训练集 类别构成的
        for docIndex in trainingSet:
            trainMat.append(self.setOfwords2vec(vocablist, docList[docIndex]))
            # 将类别添加到训练集类别标签
            trainClasses.append(classList[docIndex])

        p0V, p1V, pSpam = self.trainNB(np.array(trainMat), np.array(trainClasses))
        # 错误分类计数
        errorCount = 0

        # 遍历测试集
        for docIndex in testSet:
            # 依次返回测试集的文档向量
            wordVector = self.setOfwords2vec(vocablist, docList[docIndex])
            if self.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
                print(f"分类错误的测试集：{docList[docIndex]}")
            print("错误率：%.2f%%" % (float(errorCount) / len(testSet) * 100))


a = NavieBayes()
a.spamtest()
