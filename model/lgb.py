# -*- coding: utf-8 -*
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from model.load_data import load_data, split_test_data
import matplotlib.pylab as plt


def run_lgb(train_X, train_y, val_X, val_y, test_X, test_y):
    params = {
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

    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_val = lgb.Dataset(val_X, val_y, reference=lgb_train)
    model = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_val, early_stopping_rounds=50)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    print('ROC_AUC_SCORE:', roc_auc_score(test_y, pred_test_y))
    print('classification_report:\n', classification_report(test_y, pred_test_y))
    return pred_test_y, model

if __name__ == '__main__':
    allData = load_data()
    train_xy, test_xy = train_test_split(allData, test_size=0.2, random_state=1995)
    train, val = train_test_split(train_xy, test_size=0.2, random_state=1995)

    # use_cols = ['EMPLOYETYPE', 'HYZK', 'JOB', 'JOB_TITLE', 'MARRIAGE_STATE', 'OCCUPATIONID', 'XINGBIE',
    #             'XUELI', 'ZHICHEN', 'ZHIYE', 'max(DQQC)', 'PURPOSEID', 'JTYSR', 'PJYSR', 'HTDKJE', 'JKHTLL', 'MONTHRETURNAMOUNT', 'PUNISHRATE', 'FWZJ', 'GFSFK', 'UNITPRICE',
    #                'ALREADYPAYRATE', 'DKFFE', 'DKYE', 'HSBJZE', 'HSLXZE', 'FXZE', 'TQGHBJZE', 'YQBJZE', 'YQLXZE', 'LJYQQS',
    #                'DKQS', 'LLFDBL', 'YQQC', 'YQBJ', 'SSYQBJJE', 'SSYQFXJE', 'SSYQLXJE']

    train_X = train.ix[:, 0:-1]
    val_X = val.ix[:, 0:-1]
    test_X = test_xy.ix[:, 0:-1]

    train_y = train.ix[:, -1]
    val_y = val.ix[:, -1]
    test_y = test_xy.ix[:, -1]

    pred_test, model = run_lgb(train_X, train_y, val_X, val_y, test_X, test_y)
    print(pred_test)
    print('The roc of prediction is:', roc_auc_score(test_y, pred_test))
    print('Feature names:', model.feature_name())

    lgb.plot_importance(model, max_num_features=30)
    plt.title("Featurertances")
    # plt.show()

