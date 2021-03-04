#! /usr/bin/env python3

import numpy as np


class Data:
    def sin(self, data_points):
        x = np.linspace(0, 40, data_points).reshape((data_points, 1))
        y = 2 + 0.3*np.sin(x) + 0.1*np.random.randn(data_points, 1)
        return x, y


    def linear(self, data_points):
        x = np.linspace(0, 40, data_points).reshape((data_points, 1))
        y = 2 * x + 3 + 2*np.random.randn(data_points, 1)
        return x, y
