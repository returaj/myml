#! /usr/bin/env python3


class BaseModel:
    def train(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

