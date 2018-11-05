# -*- coding: utf-8 -*
from model.load_data import load_data, split_test_data, get_folds
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

'''
类别不平衡下的评估指标
ROC曲线下的面积（AUC_ROC）
mean Average Precesion（mAP），指的是在不同召回下的最大精确度的平均值
'''
if __name__ == '__main__':
    train = load_data()
    y = train['BinaryCategory']

    print(train.shape)
    # 还需要drop一些特征
    excluded_features = ['YQBJZE', 'YQLXZE', 'MONTHRETURNAMOUNT', 'DKZH', 'JKHTQDRQ', 'YDFKRQ', 'SSRQ','HTDKJE']

    train = train.drop(excluded_features, axis=1)
    print(train.shape)

    X_train, X_test, y_train, y_test = split_test_data(train)
    print(X_train.shape, X_test.shape)  # (23376, 33) (2598, 33)




    # 类型值特征
    categorical_features = [f for f in train.columns
                            if (f not in excluded_features) & (train[f].dtype == 'object')]

    # 分解类型值特征
    # for f in categorical_features:
    #     train[f], indexer = pd.factorize(train[f])
    #     test[f] = indexer.get_indexer(test[f])

    # folds = get_folds(df=train, n_splits=5)
    importances = pd.DataFrame()
    oof_reg_preds = np.zeros(train.shape[0])
    # sub_reg_preds = np.zeros(test.shape[0])

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'binary_error'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_valid,
                    early_stopping_rounds=5)

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('The roc of prediction is:', roc_auc_score(y_test, y_pred))

    print('Feature names:', gbm.feature_name())
    print('Feature importances:', list(gbm.feature_importance()))



