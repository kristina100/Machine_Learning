import numpy as np
import matplotlib.pyplot as plt


class SelfLinearRegression(object):
    def __init__(self, fit_intercept=True, solve='sgd', if_std=True, epochs=10, eta=1e-2, L1=None, L2=None):
        """
        :param fit_intercept:
        :param solve: 使用的方法
        :param if_std: 是否归一化，这里还可以使用正则化；有时间就试试；可以避免异常点对模型构建造成很大影响
        """
        self.w = None
        self.fit_intercept = fit_intercept
        self.solve = solve
        self.if_std = if_std
        if if_std:
            self.feature_mean = None
            self.feature_std = None
        self.epochs = epochs
        self.eta = eta
        self.L1 = L1
        self.L2 = L2
        # 将函数向量化
        self.myfunc = np.vectorize()

    def init_params(self, features):
        """
        初始化参数w
        :return:
        """
        self.w = np.random.random(size=(features, 1))

    def _fit_closed_solution(self, x, y):
        """
        直接求闭式解
        :param x:
        :param y:
        :return:返回w
        """
        if self.L1 is None and self.L2 is None:
            self.w = np.linalg.pinv(x).dot(y)
        elif self.L1 is None and self.L2 is not None:
            self.w = np.linalg.inv(x.T.dot(x) + self.L2 * np.eye(x.shape[1])).dot(x.T).dot(y)
        else:
            self._fit_sgd(x, y)

    def _fit_sgd(self, x, y):
        """
        随机梯度下降求解
        :param x:
        :param y:
        :return:
        """
        x_y = np.c_[x, y]
        # 更新w,b，数据一轮循环epoch而成，就打乱一次次序
        for _ in range(self.epochs):
            # 打乱列表，从全排列中随机选择一个
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0] // 1):
                batch_x_y = x_y[1 * index:1 * (index + 1)]
                batch_x = batch_x_y[:, :-1]
                batch_y = batch_x_y[:, -1:]

                dw = -2 * batch_x.T.dot(batch_y - batch_x.dot(self.w))
                # 添加正则化
                dw_reg = np.zeros(shape=(x.shape[1]-1,1))
                self.w = self.w - self.eta * dw

    def fit(self, x, y):
        # 是否归一化feature
        if self.if_std:
            self.feature_mean = np.mean(x, axis=0)
            self.feature_std = np.std(x, axis=0) + 1e-8
            x = (x - self.feature_mean) / self.feature_std
        # 是否训练bias
        if self.fit_intercept:
            x = np.c_[x, np.ones_like(y)]
        # 初始化参数
        self.init_params(x.shape[1])
        # 训练模型
        if self.solve == 'closed':
            self._fit_closed_solution(x, y)
        elif self.solve == 'sgd':
            self._fit_sgd(x, y)

    def get_params(self):
        """
        输出原始的系数
        :return: w,b
        """
        if self.fit_intercept:
            w = self.w[:-1]
            b = self.w[-1]
        else:
            w = self.w
            b = 0
        if self.if_std:
            w = w / self.feature_std.reshape(-1, 1)
            b = b - w.T.dot(self.feature_mean.reshape(-1, 1))
        return w.reshape(-1), b

    def plot_fit_boundary(self, x, y):
        """
        绘制拟合结果
        :param x:
        :param y:
        :return:
        """
        plt.scatter(x[:, 0], y)
        plt.plot(x[:, 0], self.predict(x), 'r')

        # 模型评估

    def predict(self, x):
        """
        :param x:ndarray格式数据: m x n
        :return: m x 1

        """
        if self.if_std:
            x = (x - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x = np.c_[x, np.ones(shape=x.shape[0])]

        print(type(x.dot(self.w)))
        return x.dot(self.w)

