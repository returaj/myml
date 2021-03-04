#! /usr/bin/env python3

import numpy as np
from classification.data.data import Data
from classification.model.base_model import BaseModel
from classification.model.base_runner import BaseRunner
from tqdm import tqdm


class LogisticRegression(BaseModel):
    def __init__(self):
        self.w = None
        self.alpha = 1e-1

    def initialize(self, feature_size):
        self.w = np.zeros(feature_size + 1)

    def update(self, x, y, lmbda=0.01):
        w = self.w
        grd = np.sum(x.T*y / (1 + np.exp(np.dot(x, w) * y)), axis=1) - lmbda * w
        w += self.alpha * grd

    def train(self, X, Y, epochs=10):
        batch_size, feature_size = X.shape
        self.initialize(feature_size)
        X = np.concatenate((np.ones((batch_size, 1)), X), axis=1)
        xids = list(range(batch_size))
        mini_batch_size = 10
        for e in range(1, epochs):
            np.random.shuffle(xids)
            s = 0
            while s < batch_size:
                e = min(batch_size, s+mini_batch_size)
                ids = xids[s:e]
                self.update(X[ids], Y[ids], lmbda=0.1)
                s = e

    def predict(self, x):
        wTx = self.w[0] + np.dot(self.w[1:], x)
        prob_plus_1 = 1 / (1 + np.exp(-wTx))
        prob_minus_1 = 1 / (1 + np.exp(wTx))
        return 1 if prob_plus_1 > prob_minus_1 else -1


class Runner(BaseRunner):
    def pre_process(self, X, Y):
        newY = []
        for y in Y:
            l = -1 if y==0 else 1
            newY.append(l)
        return np.array(X), np.array(newY)

    def experiment(self, data_fun, figname):
        X, Y = self.pre_process(*data_fun(n1=200, n2=200))

        print(f"Generate decision boundary for {figname}")
        log_reg = LogisticRegression()
        log_reg.train(X, Y)

        Xtest, Ytest = self.pre_process(*data_fun(n1=100, n2=100))
        print(f"Accuracy: {self.accuracy(log_reg, Xtest, Ytest)}")

        x, y = [], []
        for a, b in X:
            x.append(a); y.append(b)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        self.visualize(log_reg, (min_x, max_x), (min_y, max_y), figname)


if __name__ == '__main__':
    data = Data()
    runner = Runner()

    runner.experiment(data.circle, "classification/logistic_regression/circle.png")
    runner.experiment(data.ls_balanced, "classification/logistic_regression/ls_balanced.png")
    runner.experiment(data.ls_unbalanced, "classification/logistic_regression/ls_unbalanced.png")



