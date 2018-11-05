# -*- coding: utf-8 -*
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from model.load_data import load_data, split_test_data
from sklearn.metrics import roc_auc_score


def LR(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(random_state=1995)
    lr.fit(X_train, y_train)
    y_pred_test = lr.predict(X_test)
    # 0.4998749061796347
    print('ROC_AUC_SCORE:', roc_auc_score(y_test, y_pred_test))
    print('classification_report:\n', classification_report(y_test, y_pred_test))


if __name__ == '__main__':
    allData = load_data()
    X_train, X_test, y_train, y_test = split_test_data(allData)
    # 预处理
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    LR(X_train, X_test, y_train, y_test)

