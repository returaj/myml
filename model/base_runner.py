#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm


class BaseRunner:
    def experiment(self, data_fun, figname):
        raise NotImplementedError


    def accuracy(self, model, Xtest, Ytest):
        acc, n = 0, len(Ytest)
        for x, y in zip(Xtest, Ytest):
            acc += 1 if model.predict(x)==y else 0
        return acc / n

    @staticmethod
    def visualize(model, x_range, y_range, figname):
        x = np.arange(*x_range, 0.1)
        y = np.arange(*y_range, 0.1)
        xx, yy = np.meshgrid(x, y)
        fx, fy = xx.flatten(), yy.flatten()
        X = np.column_stack((fx, fy))
        Yhat = []
        for x in tqdm(X):
            Yhat.append(model.predict(x))
        zz = np.array(Yhat).reshape(xx.shape)

        plt.contourf(xx, yy, zz, cmap='Paired')
        plt.savefig(figname)

