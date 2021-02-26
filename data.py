#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Data:

    def circle(self, n1, n2):
        def circle(r):
            x = np.random.uniform(-r, r)
            sign = np.random.choice([-1, 1])
            y = np.sqrt(r*r - x*x) * sign
            return x, y

        X1, X2 = [], []
        r1, r2 = 1, 3
        for i in range(n1):
            x, y = circle(r1)
            X1.append((x, y))
        for i in range(n2):
            x, y = circle(r2)
            X2.append((x, y))
        return X1, X2

    def xor(self):
        X1 = [(0, 0), (1, 1)]
        X2 = [(0, 1), (1, 0)]
        return X1, X2

    def ls_balanced(self, n1, n2):
        X1, X2 = [], []
        x1, y1 = (5, 10), (-5, 5)
        for i in range(n1):
            x, y = np.random.uniform(*x1), np.random.uniform(*y1)
            X1.append((x, y))
        x2, y2 = (-2, 3), (-5, 5)
        for i in range(n2):
            x, y = np.random.uniform(*x2), np.random.uniform(*y2)
            X2.append((x, y))
        return X1, X2

    def ls_unbalanced(self, n1, n2):
        X1, X2 = [], []
        x1, y1 = (7, 10), (-5, 5)
        for i in range(n1):
            x, y = np.random.uniform(*x1), np.random.uniform(*y1)
            X1.append((x, y))
        extreme_n2 = 3
        ex2, ey2 = (5, 6), (-3, 3)
        for i in range(extreme_n2):
            x, y = np.random.uniform(*ex2), np.random.uniform(*ey2)
            X2.append((x, y))
        clusture_n2 = n2 - extreme_n2
        cx2, cy2 = (-3, 1), (-5, 5)
        for i in range(clusture_n2):
            x, y = np.random.uniform(*cx2), np.random.uniform(*cy2)
            X2.append((x, y))
        return X1, X2


def visualize(X1, X2, title, figname):
    x1, y1 = [], []
    for x, y in X1:
        x1.append(x)
        y1.append(y)

    x2, y2 = [], []
    for x, y in X2:
        x2.append(x)
        y2.append(y)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x1, y1, 'o')
    ax.plot(x2, y2, 'x')
    plt.title(title)
    plt.savefig(figname)


if __name__ == '__main__':
    n1, n2 = 20, 20

    data = Data()

    circleX1, circleX2 = data.circle(n1, n2)
    visualize(circleX1, circleX2, "circle data", "circle.png")

    lsbX1, lsbX2 = data.ls_balanced(n1, n2)
    visualize(lsbX1, lsbX2, "linear separable balanced", "ls_balanced.png")

    lsuX1, lsuX2 = data.ls_unbalanced(n1, n2)
    visualize(lsuX1, lsuX2, "linear separable unbalanced", "ls_unbalanced.png")



