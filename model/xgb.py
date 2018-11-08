# -*- coding: utf-8 -*
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model.load_data import load_data, split_test_data


def run_xgb(X_train, y_train, X_val, y_val, X_test, y_test):
    gbm = xgb.XGBClassifier(
        # learning_rate = 0.02,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=2,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1).fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    print("xgboost准确率为：", gbm.score(X_test, y_test))


if __name__ == '__main__':
    allData = load_data()
    train_xy, test_xy = train_test_split(allData, test_size=0.2, random_state=1995)
    train, val = train_test_split(train_xy, test_size=0.2, random_state=1995)

    X_train = train.ix[:, 0:-1]
    X_val = val.ix[:, 0:-1]
    X_test = test_xy.ix[:, 0:-1]

    y_train = train.ix[:, -1]
    y_val = val.ix[:, -1]
    y_test = test_xy.ix[:, -1]

    run_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
