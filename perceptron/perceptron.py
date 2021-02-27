#! /usr/bin/env python3

import numpy as np
from data.data import Data
from model.base_model import BaseModel
from model.base_runner import BaseRunner


class Perceptron(BaseModel):
    def __init__(self):
        self.w = None

    def train(self, X, Y, max_runs=1000):
        batch, dim = X.shape
        X = np.concatenate((np.ones((batch, 1)), X), axis=1)  # add column of all ones in X: add bias
        w = np.zeros(dim+1)
        m = batch; runs = 0
        while m > 0 and runs < max_runs:
            m = 0
            for x, y in zip(X, Y):
                err = y * np.dot(x, w)
                if err <= 0:
                    w += y * x
                    m += 1
            runs += 1
        self.w = w

    def predict(self, x):
        pred = np.dot(self.w[1:], x) + self.w[0]
        return 1 if pred > 0 else -1


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
        perceptron = Perceptron()
        perceptron.train(X, Y)

        x, y = [], []
        for a, b in X:
            x.append(a); y.append(b)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        self.visualize(perceptron, (min_x, max_x), (min_y, max_y), figname)


if __name__ == '__main__':
    data = Data()
    runner = Runner()

    runner.experiment(data.circle, "perceptron/circle.png")
    runner.experiment(data.ls_balanced, "perceptron/ls_balanced.png")
    runner.experiment(data.ls_unbalanced, "perceptron/ls_unbalanced.png")
    runner.experiment(data.xor, "perceptron/xor.png")





