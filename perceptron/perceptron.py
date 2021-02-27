#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data.data import Data
from model.base_model import BaseModel


class Perceptron(BaseModel):
    def __init__(self):
        self.w = None

    def train(self, X, Y, max_runs=10000):
        batch, dim = X.shape
        X = np.concatenate((np.ones((batch, 1)), X), axis=1)  # add column of all ones in X: add bias
        w = np.zeros(dim+1)
        m = batch; runs = 0
        while m > 0 and runs < max_runs:
            m = 0
            for x, y in zip(X, Y):
                err = y * np.dot(x, w)
                if err <= 0:
                    w += err
                    m += 1
            runs += 1
        self.w = w

    def predict(self, x):
        return 1 if np.dot(self.w, x) > 0 else -1



