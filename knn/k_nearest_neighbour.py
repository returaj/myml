#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from data.data import Data
from model.base_model import BaseModel


class KNN(BaseModel):
    def __init__(self, k, dist):
        self.k = k
        self.dist = dist

    def train(self, X, Y):
        self.X = X
        self.Y = Y

    @staticmethod
    def vote(d, k):
        y = defaultdict(lambda : 0)
        max_v = 0
        for i in range(min(k, len(d))):
            _, label = d[i]
            y[label] += 1
            max_v = max(max_v, y[label])
        labels_set = [k for k, v in y.items() if max_v == v]
        if len(labels_set) > 1:
            return self.vote(d, k-2)
        return labels_set[0]

    def predict(self, x):
        d = []
        for dx, dy in zip(self.X, self.Y):
            d.append((self.dist(x, dx), dy))
        d.sort(key=lambda x: x[0])
        return self.vote(d, self.k)


def distance_func(p):
    def dist(x1, x2):
        diff = np.abs(x1-x2)
        max_v = np.max(diff)
        if p > 6 or (max_v > 50 and p > 3):
            return max_v
        return np.power(np.sum(np.power(diff, p)), 1/p)
    return dist


def cosine_fun():
    return lambda x1, x2: np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def test_accuracy(knn, X, Y):
    acc = 0
    for x, y in zip(X, Y):
        acc += 1 if y == knn.predict(np.array(x)) else 0
    return acc / len(X)


def visualize(knn, x_range, y_range, figname):
    x = np.arange(*x_range, 0.1)
    y = np.arange(*y_range, 0.1)
    xx, yy = np.meshgrid(x, y)
    fx, fy = xx.flatten(), yy.flatten()
    Xhat = np.column_stack((fx, fy))
    Yhat = []
    for x in tqdm(Xhat):
        Yhat.append(knn.predict(x))
    zz = np.array(Yhat).reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap='Paired')
    plt.savefig(figname)


def experiment(data_func, figname):
    k = 5
    p = 2

    print(f"Generate KNN boundary in {figname}")
    X, Y = data_func(n1=200, n2=200)

    knn = KNN(k, distance_func(p))
    knn.train(X, Y)

    x, y = [], []
    for a, b in X:
        x.append(a); y.append(b)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    visualize(knn, (min_x, max_x), (min_y, max_y), figname)


if __name__ == '__main__':
    data = Data()
    experiment(data.circle, "knn/circle.png")
    experiment(data.ls_balanced, "knn/ls_balanced.png")
    experiment(data.ls_unbalanced, "knn/ls_unbalanced.png")



