# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from model.load_data import load_data, split_test_data


def IForest(data):
    clf = IsolationForest(max_samples=100, random_state=1995)
    clf.fit(data)
    y_pred_train = clf.predict(data)
    return y_pred_train


def show_diff(y_pred_train, y_train):
    count = 0
    for i in range(y_train.shape[0]):
        if y_pred_train[i] != y_train[i]:
            count += 1

    return count, y_train.shape[0]


if __name__ == '__main__':
    allData = load_data()
    X_train, X_test, y_train, y_test = split_test_data(allData)
    print("train集X大小为：", X_train.shape)
    print("test集X大小为：", X_test.shape)
    print("train集y大小为：", y_train.shape)
    print("test集y大小为：", y_test.shape)
    y_pred_train = IForest(X_train)
    print(y_pred_train)
    print(show_diff(y_pred_train, y_train))

