import numpy as np


class SelfLogisticRegression(object):
    def __init__(self, fit_intercept=True, solver='sgd', if_standard=True, L1=None, L2=None, epochs=10,
                 eta=None, batch_size=16):

        self.w = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.if_standard = if_standard
        if if_standard:
            self.feature_mean = None
            self.feature_std = None
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.L1 = L1
        self.L2 = L2

        # Record loss function
        self.losses = []

    def Sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def init_params(self, n_features):
        """
        Initialization parameters
        :return:
        """
        self.w = np.random.random(size=(n_features, 1))

    def _fit_closed_solution(self, x, y):
        """
        Direct closed-form solution
        :param x:
        :param y:
        :return:
        """
        self._fit_sgd(x, y)

    def _fit_sgd(self, x, y):
        """
         Stochastic gradient descent solver
        :param x:
        :param y:
        :return:
        """
        x_y = np.c_[x, y]
        count = 0
        for _ in range(self.epochs):
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0] // self.batch_size):
                count += 1
                batch_x_y = x_y[self.batch_size * index:self.batch_size * (index + 1)]
                batch_x = batch_x_y[:, :-1]
                batch_y = batch_x_y[:, -1:]

                dw = -1 * (batch_y - self.Sigmoid(batch_x.dot(self.w))).T.dot(batch_x) / self.batch_size
                dw = dw.T

                # Add Ridge Regression and Lasso Regression
                dw_reg = np.zeros(shape=(x.shape[1] - 1, 1))
                if self.L1 is not None:
                    dw_reg += self.L1 * np.vectorize(self.w[:-1]) / self.batch_size
                if self.L2 is not None:
                    dw_reg += 2 * self.L2 * self.w[:-1] / self.batch_size
                dw_reg = np.concatenate([dw_reg, np.asarray([[0]])], axis=0)

                dw += dw_reg
                self.w = self.w - self.eta * dw

            # losses
            cost = -1 * np.sum(
                np.multiply(y, np.log(self.Sigmoid(x.dot(self.w)))) + np.multiply(1 - y, np.log(
                    1 - self.Sigmoid(x.dot(self.w)))))
            self.losses.append(cost)

    def fit(self, x, y):
        """
        :param x: ndarray: m x n
        :param y: ndarray: m x 1
        :return:
        """
        y = y.reshape(x.shape[0], 1)
        # feature normalized or not
        if self.if_standard:
            self.feature_mean = np.mean(x, axis=0)
            self.feature_std = np.std(x, axis=0) + 1e-8
            x = (x - self.feature_mean) / self.feature_std
        # Whether to train bias
        if self.fit_intercept:
            x = np.c_[x, np.ones_like(y)]
        # Initialization parameters
        self.init_params(x.shape[1])
        # Update eta
        if self.eta is None:
            self.eta = self.batch_size / np.sqrt(x.shape[0])

        if self.solver == 'closed':
            self._fit_closed_solution(x, y)
        elif self.solver == 'sgd':
            self._fit_sgd(x, y)

    def get_params(self):
        """
        Output the original coefficients
        :return: w,b
        """
        if self.fit_intercept:
            w = self.w[:-1]
            b = self.w[-1]
        else:
            w = self.w
            b = 0
        if self.if_standard:
            w = w / self.feature_std.reshape(-1, 1)
            b = b - w.T.dot(self.feature_mean.reshape(-1, 1))
        return w.reshape(-1), b

    def predict_proba(self, x):
        """
        The probability that the prediction is y = 1
        :param x:ndarray: m x n
        :return: m x 1
        """
        if self.if_standard:
            x = (x - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x = np.c_[x, np.ones(x.shape[0])]
        return self.Sigmoid(x.dot(self.w))

    def predict(self, x):
        """
        Prediction category, default is 1 for greater than 0.5, 0 for less than 0.5
        :param x:
        :return:
        """
        proba = self.predict_proba(x)
        return (proba > 0.5).astype(int)

