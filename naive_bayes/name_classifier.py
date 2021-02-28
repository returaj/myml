#! /usr/bin/env python3

# Accuracy of 71.5 %

import string
from hashlib import md5
import numpy as np
import os
from naive_bayes import NaiveBayes, Categorical
from tqdm import tqdm


FEATURES_SIZE = 83

K = 3

N = 1000


def select_name(filepath):
    names = []
    with open(filepath, 'r') as fp:
        for line in fp:
            names.append(line.strip().lower())
    return names


def hash(s):
    return int(md5(s.encode()).hexdigest(), 16) % FEATURES_SIZE


def find_all_substrings(name, k):
    arr = list(name)[-3:]
    substrings = []
    for i in range(len(arr)-k):
        j, sub = 0, ''
        while j < k:
            sub += arr[i+j]
            j += 1
        substrings.append(hash(sub))
    return substrings


def to_feature(name):
    X = np.zeros(FEATURES_SIZE)
    for i in range(1, K+1):
        substrings = find_all_substrings(name, i)
        for fid in substrings:
            X[fid] = 1
    return X


def pre_process(names):
    X = []
    for n in range(len(names)):
        X.append(to_feature(names[n]))
    return np.array(X)


def test_accuracy(model, X, Y):
    acc = 0
    for xname, y in zip(X, Y):
        if model.predict(to_feature(xname)) == y:
            acc += 1
    return acc / len(X)


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(path + "/../data")
    male_path = os.path.abspath(path + "/boy_firstname.txt")
    female_path = os.path.abspath(path + "/female_firstname.txt")

    runs, acc = 30, 0

    ftypes = []
    for fi in range(FEATURES_SIZE):
        c = Categorical()
        c.set_elements([0, 1])
        ftypes.append(c)

    male_names = select_name(male_path)
    female_names = select_name(female_path)

    for r in tqdm(range(1, runs+1)):
        np.random.shuffle(male_names)
        train_male, test_male = male_names[0:N], male_names[N::]
        X_male = pre_process(train_male)
        Y_male = ["male"] * len(train_male)

        np.random.shuffle(female_names)
        train_female, test_female = female_names[0:N], female_names[N::]
        X_female = pre_process(train_female)
        Y_female = ["female"] * len(train_female)

        X = np.row_stack((X_male, X_female))
        Y = Y_male + Y_female

        nb = NaiveBayes()
        nb.train(X, Y, ftypes=ftypes)

        test_names = test_male + test_female
        test_Y = ["male"] * len(test_male) + ["female"] * len(test_female)

        acc += (test_accuracy(nb, test_names, test_Y) - acc) / r

    print(acc)

    while True:
        name = input("Enter test name: ")
        print(nb.predict(to_feature(name.strip().lower())))


if __name__ == '__main__':
    main()




