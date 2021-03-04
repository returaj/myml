#! /usr/bin/env python3

import numpy as np
from collections import defaultdict
from classification.data.data import Data
from classification.model.base_model import BaseModel
from classification.model.base_runner import BaseRunner


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
        k = min(k, len(d))
        k = k-1 if k%2==0 else k
        for i in range(k):
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


class Runner(BaseRunner):
    def experiment(self, data_func, figname):
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
        self.visualize(knn, (min_x, max_x), (min_y, max_y), figname)


if __name__ == '__main__':
    data = Data()
    runner = Runner()
    runner.experiment(data.circle, "classification/knn/circle.png")
    runner.experiment(data.ls_balanced, "classification/knn/ls_balanced.png")
    runner.experiment(data.ls_unbalanced, "classification/knn/ls_unbalanced.png")
    runner.experiment(data.xor, "classification/knn/xor.png")


