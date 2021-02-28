#! /usr/bin/env python3

import numpy as np
from model.base_model import BaseModel


class MultinomialNaiveBayes(BaseModel):
    def set_features_size(self, fsize):
        self.fsize = fsize

    def set_labels(self, labels):
        labels = list(set(labels))
        self.labels_to_id = {l: i for i, l in enumerate(labels)}
        self.id_to_labels = {i: l for i, l in enumerate(labels)}
        self.prob_labels = np.ones(len(labels))

    def initialize(self):
        fsize, lsize = self.fsize, len(self.labels_to_id)
        self.prob_feature = np.ones((fsize, lsize))

    def update(self):
        self.prob_labels /= np.sum(self.prob_labels, axis=0)
        self.prob_feature /= np.sum(self.prob_feature)

    def train(self, X, Y, fsize=None):
        if fsize is not None:
            self.set_features_size(fsize)
        self.set_labels(Y)
        self.initialize()

        for x, y in zip(X, Y):
            yid = self.labels_to_id[y]
            self.prob_labels[yid] += 1
            for fi in range(self.fsize):
                self.prob_feature[fi, yid] += x[fi]
        self.update()

    def prob(self, x, yid):
        p = np.log(self.prob_labels[yid])
        for fi in range(self.fsize):
            p += x[fi] * np.log(self.prob_feature[fi, yid])
        return p

    def predict(self, x):
        yprob = []
        for yid in range(len(self.id_to_labels)):
            yprob.append(self.prob(x, yid))
        y, max_v = 0, yprob[0]
        for yid in range(len(yprob)):
            if yprob[yid] > max_v:
                y, max_v = yid, yprob[yid]
        return self.id_to_labels[y]




