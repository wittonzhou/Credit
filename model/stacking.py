# -*- coding: utf-8 -*
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import lightgbm as lgb
import numpy as np

from model.load_data import load_data, split_test_data


def run(X_dev, X_test, y_dev, y_test):

    lightgbm_params = {
        "objective": "binary",
        "metric": {'l2', 'auc'},
        "num_leaves": 16,
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_freq": 5,
        "verbosity": 0,
        'seed': 1995
    }

    clfs = [
        RandomForestClassifier(random_state=1995),
        XGBClassifier(n_estimators=1000, max_depth=4,
        min_child_weight=2,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic'),

        lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            num_leaves=16,
            learning_rate=0.05,
            bagging_fraction=0.7,
            feature_fraction=0.7,
            bagging_freq=5,
            seed=1995,
            n_estimators=200),
    ]
    # 20%作为测试集， 80%作为训练集
    # dev_cutoff = len(y) * 4 / 5
    # X_dev = X[:dev_cutoff]
    # y_dev = y[:dev_cutoff]
    # X_test = X[dev_cutoff:]
    # y_test = y[dev_cutoff:]

    # ready for cross validation
    skf = StratifiedKFold(n_splits=5)

    blend_train = np.zeros((X_dev.shape[0], len(clfs)))  # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs)))  # Number of testing data x Number of classifiers

    for j, clf in enumerate(clfs):
        print("Training classifier %s" % j)
        blend_test_j = np.zeros((X_test.shape[0], 5))  # Number of testing data x Number of folds , we will take the mean of the predictions later
        i = 0
        for train_index, cv_index in skf.split(X_dev, y_dev):
            print("Fold %s" % i)

            # This is the training and validation set
            X_train = X_dev[train_index]
            y_train = y_dev[train_index]
            X_cv = X_dev[cv_index]
            y_cv = y_dev[cv_index]

            clf.fit(X_train, y_train)
            y_pred_cv = clf.predict(X_cv)
            blend_train[cv_index, j] = y_pred_cv
            blend_test_j[:, i] = clf.predict(X_test)
            i += 1
            if i >= 5:
                break

        blend_test[:, j] = blend_test_j.mean(1)

    print("y_dev.shape = %s" % y_dev.shape)

    # start blending
    bclf = LogisticRegression()
    bclf.fit(blend_train, y_dev)

    # predict now
    y_test_predict = bclf.predict(blend_test)
    score = accuracy_score(y_test, y_test_predict)

    print('ROC_AUC_SCORE: %s' % roc_auc_score(y_test, y_test_predict))
    print('Accuracy = %s' % score)


if __name__ == '__main__':
    allData = load_data()
    X_train, X_test, y_train, y_test = split_test_data(allData)
    run(X_train, X_test, y_train, y_test)
