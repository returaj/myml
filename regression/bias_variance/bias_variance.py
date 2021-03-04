#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm


class LinearRegression:
    def __init__(self):
        # y = m*x + c + d * sin(x)
        self.m, self.c, self.d = 2, 3, 0.01
        # weight update alpha
        self.alpha = 1e-3

    def data(self, data_points=10):
        x = np.random.uniform(1, 30, data_points).reshape((data_points, 1))
        y = self.m * x + self.c + self.d*np.sin(x) + 0.1*np.random.randn(data_points, 1)
        return x, y

    def optimal_y(self, X):
        opti_y = self.m * X + self.c + self.d * np.sin(X)
        return opti_y

    def get_err(self, w, D):
        X, Y = D
        y_pred = self.get_predicted(w, X)
        err = (Y - y_pred) ** 2
        return np.sum(err) / len(X)

    def train(self, D, lmbda, epochs=500):
        X, Y = D
        bsize, fsize = X.shape
        w = np.zeros((fsize+1, 1))
        mini_batch = 5
        xids = list(range(bsize))
        for ep in range(1, epochs+1):
            np.random.shuffle(xids)
            s = 0
            while s < bsize:
                e = min(s+mini_batch, bsize)
                ids = xids[s:e]
                x, y = X[ids], Y[ids]
                y_pred = self.get_predicted(w, x)
                assert y_pred.shape == (e-s, 1)
                # update
                grdW = np.sum((y - y_pred) * x, axis=0, keepdims=True) / (e-s) - lmbda * w[1:].T
                grdB = np.sum(y-y_pred) - lmbda * w[0, 0]
                w[1:] += self.alpha * grdW.T
                w[0, 0] += self.alpha *  grdB
                s = e
        return w

    def get_predicted(self, w, X):
        return w[0, 0] + np.dot(X, w[1:])

    def bias_variance(self, lmbda, num_datas=200):
        w_ = np.zeros((2, 1))
        actual_err = 0
        variance, noise, bias = 0, 0, 0
        print(f"bias_variance evaluation for lambda: {lmbda}")
        for d in tqdm(range(1, num_datas+1)):
            w = self.train(self.data(), lmbda)

            testx, testy = self.data(1)
            actual_err += (self.get_err(w, (testx, testy)) - actual_err) / d
            dov = self.get_predicted(w, testx) - self.get_predicted(w_, testx)
            dob = self.optimal_y(testx) - self.get_predicted(w_, testx)
            w_ += (w - w_) / d
            dnv = self.get_predicted(w, testx) - self.get_predicted(w_, testx)
            dnb = self.optimal_y(testx) - self.get_predicted(w_, testx)
            variance += np.sum(dov * dnv)
            bias += np.sum(dob * dnb)

            dn = self.optimal_y(testx) - testy
            noise += np.sum(dn**2)
        print(w_)

        return actual_err, variance / num_datas, bias / num_datas, noise / num_datas

    def evaluate(self, figname):
        lmbdas = 0.1 * np.linspace(0, 10, 11)
        err, sum_err = [], []
        var, bias, noise = [], [], []
        for l in lmbdas:
            e, v, b, n = self.bias_variance(l)
            se = v+b+n
            sum_err.append(se)
            err.append(e); var.append(v); bias.append(b); noise.append(n)

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(lmbdas, err, 'r-', label="actual error")
        ax.plot(lmbdas, sum_err, 'b-', label="sum error")
        ax.plot(lmbdas, var, 'k-', label="variance error")
        ax.plot(lmbdas, bias, 'y-', label="bias error")
        ax.plot(lmbdas, noise, 'g-', label="noise error")

        plt.legend()
        plt.savefig(figname)


if __name__ == '__main__':
    lr = LinearRegression()
    lr.evaluate('regression/bias_variance/curve.png')


