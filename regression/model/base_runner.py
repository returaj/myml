#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class BaseRunner:
    def experiment(self, data_fun, figname):
        raise NotImplementedError

    @staticmethod
    def visualize(model, X, Y, figname):
        xmin, xmax = np.min(X), np.max(X)
        Xtest = np.linspace(xmin, xmax, 50).reshape((50, 1))
        Ytest = model.predict(Xtest)
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(Xtest, Ytest, 'k-', label="predicted output")
        ax.plot(X, Y, '*', label="data points")

        plt.legend()
        plt.savefig(figname)

