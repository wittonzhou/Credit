# -*- coding: utf-8 -*
import numpy as np
from sklearn import svm

from model.load_data import load_data, split_test_data
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
import pandas as pd


def oneClassSVMClassify(X_majroity, X_train, X_test, y_train, y_test):
    clf = svm.OneClassSVM(nu=0.1, gamma=0.1,random_state=1995)
    clf.fit(X_majroity)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(classification_report(y_train, y_pred_train))
    print(classification_report(y_test, y_pred_test))

    print('ROC_AUC_SCORE:', roc_auc_score(y_test, y_pred_test))
    # 考虑类别不平衡，用weighted
    print('train dataset F1_SCORE:', f1_score(y_train, y_pred_train, average='weighted'))
    print('test dataset F1_SCORE:', f1_score(y_test, y_pred_test, average='weighted'))
    print('train dataset accuracy_score:', accuracy_score(y_train, y_pred_train))
    print('test dataset accuracy_score:', accuracy_score(y_test, y_pred_test))


def get_maj_min_data(df=None):
    majroity = df[df.binaryCategory == 1]
    minroity = df[df.binaryCategory == -1]

    X_majroity = majroity.ix[:, 0:-1]
    X_minroity = minroity.ix[:, 0:-1]
    y_majority = majroity.ix[:, -1]
    y_minroity = minroity.ix[:, -1]
    print(y_majority.shape)
    print(y_minroity.shape)
    return X_majroity, X_minroity, y_majority, y_minroity

if __name__ == '__main__':
    allData = load_data()
    X_train, X_test, y_train, y_test = split_test_data(allData)
    X_train = pd.DataFrame(X_train, columns=['EMPLOYETYPE', 'HYZK', 'JOB', 'JTYSR', 'PJYSR', 'XINGBIE', 'XUELI',
       'ZHICHEN', 'ZHIYE', 'PURPOSEID', 'FWZJ', 'GFSFK', 'UNITPRICE','ALREADYPAYRATE', 'DKFFE'])
    y_train = pd.DataFrame(y_train, columns=['binaryCategory'])
    train = pd.concat([X_train, y_train], axis=1)
    X_majroity, X_minroity, y_majority, y_minroity = get_maj_min_data(train)

    oneClassSVMClassify(X_majroity, X_train, X_test, y_train, y_test)

