#! /usr/bin/env python3

import numpy as np
from regression.data.data import Data
from regression.model.base_model import BaseModel
from regression.model.base_runner import BaseRunner


class LinearRegression(BaseModel):
    def __init__(self):
        self.w = None
        self.alpha = 1e-3

    def initialize(self, feature_size):
        self.w = np.zeros((feature_size+1, 1))
        self.feature_size = feature_size

    def update(self, x, y, y_pred, lmbda=0.01):
        grd = np.sum((y - y_pred) * x, axis=0, keepdims=True) / len(x) - lmbda * self.w.T
        assert grd.shape == (1, self.feature_size+1)
        self.w += self.alpha * grd.T

    def get_err(self, X, Y):
        y_pred = np.dot(X, self.w)
        err = (Y - y_pred)
        return (np.dot(err.T, err)[0,0]) / len(X)

    def train(self, X, Y, delta=0.1, epochs=50):
        batch_size, feature_size = X.shape
        X = np.concatenate((np.ones((batch_size, 1)), X), axis=1)
        self.initialize(feature_size)

        mini_batch_size = 5
        xids = list(range(batch_size))
        err = 0
        for ep in range(1, epochs+1):
            np.random.shuffle(xids)
            s = 0; 
            while s < batch_size:
                e = min(batch_size, s+mini_batch_size)
                ids = xids[s:e]
                x, y = X[ids], Y[ids]
                y_pred = np.dot(x, self.w)
                self.update(x, y, y_pred)
                s = e
            if ep % 100 == 0:
                err = self.get_err(X, Y)
                print(f"Training Error after {ep} epochs: {err}\r", end="")
                if err < delta:
                    break
        print()

    def predict(self, X):
        batch_size, feature_size = X.shape
        X = np.concatenate((np.ones((batch_size, 1)), X), axis=1)
        return np.dot(X, self.w)


class Runner(BaseRunner):
    def experiment(self, data_fun, figname):
        X, Y = data_fun(100)

        lr = LinearRegression()
        lr.train(X, Y, epochs=1000)

        self.visualize(lr, X, Y, figname)


if __name__ == '__main__':
    data = Data()
    runner = Runner()

    runner.experiment(data.sin, "regression/linear_regression/sin.png")
    runner.experiment(data.linear, "regression/linear_regression/linear.png")



