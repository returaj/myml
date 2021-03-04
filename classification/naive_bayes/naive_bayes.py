#! /usr/bin/env python3

import numpy as np
from classification.data.data import Data
from classification.model.base_model import BaseModel
from classification.model.base_runner import BaseRunner


class Feature:
    def set_labels(self, labels):
        self.labels_to_id = {l: i for i, l in enumerate(labels)}

    def update(self, x, y):
        raise NotImplementedError

    def prob(self, x):
        raise NotImplementedError


class Categorical(Feature):
    def __init__(self):
        self.w = None
        self.is_normalized = False

    def set_elements(self, elements):
        self.elements_to_id = {e: i for i, e in enumerate(elements)}

    def initialize(self):
        num_elems, num_labels = len(self.elements_to_id), len(self.labels_to_id)
        self.w = np.ones((num_elems, num_labels))

    def update(self, x, y):
        if self.w is None:
            self.initialize()
        xid = self.elements_to_id[x]
        yid = self.labels_to_id[y]
        self.w[xid, yid] += 1

    def normalize(self):
        self.w /= np.sum(self.w, axis=0)
        self.is_normalized = True

    def prob(self, x, y):
        if not self.is_normalized:
            self.normalize()
        xid = self.elements_to_id[x]
        yid = self.labels_to_id[y]
        return self.w[xid, yid]


class Continuous(Feature):
    def __init__(self):
        self.mu, self.var = None, None
        self.total = None
        self.is_normalized = False

    def initialize(self):
        num_labels = len(self.labels_to_id)
        self.mu = np.zeros(num_labels)
        self.var = np.zeros(num_labels)
        self.total = np.zeros(num_labels)

    def update(self, x, y):
        if (self.mu is None) or (self.var is None):
            self.initialize()
        id = self.labels_to_id[y]
        self.total[id] += 1
        delta = (x - self.mu[id])
        self.mu[id] += delta / self.total[id] 
        self.var[id] += delta * (x - self.mu[id])

    def normalize(self):
        self.var[self.total < 2] = 0.01
        ids = self.total >= 2
        self.var[ids] /= self.total[ids]
        self.is_normalized = True

    def gaussian(self, x, m, v):
        return np.exp(-(x-m)**2 / (2*v)) / np.sqrt(2 * np.pi * v) 

    def prob(self, x, y):
        if not self.is_normalized:
            self.normalize()
        id = self.labels_to_id[y]
        m, v = self.mu[id], self.var[id]
        return self.gaussian(x, m, v)


class NaiveBayes(BaseModel):
    def __init__(self):
        self.ftypes = None

    def set_feature_types(self, ftypes):
        self.ftypes = ftypes

    def set_labels(self, labels):
        labels = list(set(labels))
        self.labels_to_id = {l: i for i, l in enumerate(labels)}
        self.id_to_labels = {i: l for i, l in enumerate(labels)}
        self.prob_labels = np.ones(len(labels))
        for f in self.ftypes:
            f.set_labels(labels)

    def train(self, X, Y, ftypes=None):
        if ftypes is not None:
            self.set_feature_types(ftypes)
        self.set_labels(Y)
        batch, num_features = X.shape
        for x, y in zip(X, Y):
            yid = self.labels_to_id[y]
            self.prob_labels[yid] += 1
            for fi in range(num_features):
                self.ftypes[fi].update(x[fi], y)
        self.prob_labels /= np.sum(self.prob_labels)

    def predict(self, x):
        y_pred = [None] * len(self.id_to_labels)
        for i, l in self.id_to_labels.items():
            p = np.log(self.prob_labels[i])
            for ft, fx in zip(self.ftypes, x):
                p += np.log(ft.prob(fx, l))
            y_pred[i] = p
        yid, max_v = 0, y_pred[0]
        for i in range(len(y_pred)):
            if max_v < y_pred[i]:
                max_v = y_pred[i]
                yid = i
        return self.id_to_labels[yid]


class Runner(BaseRunner):
    def experiment(self, data_fun, figname):
        nb = NaiveBayes()

        X, Y = data_fun(n1=200, n2=200)
        ftypes = [Continuous(), Continuous()]
        nb.train(X, Y, ftypes=ftypes)

        x, y = [], []
        for a, b in X:
            x.append(a); y.append(b)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        self.visualize(nb, (min_x, max_x), (min_y, max_y), figname)


if __name__ == '__main__':
    data = Data()
    runner = Runner()

    runner.experiment(data.circle, "classification/naive_bayes/circle.png")
    runner.experiment(data.ls_balanced, "classification/naive_bayes/ls_balanced.png")
    runner.experiment(data.ls_unbalanced, "classification/naive_bayes/ls_unbalanced.png")
    runner.experiment(data.xor, "classification/naive_bayes/xor.png")



