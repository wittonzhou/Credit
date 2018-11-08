# -*- coding: utf-8 -*
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from model.load_data import load_data, split_test_data
from sklearn.metrics import roc_auc_score


def LR(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(random_state=1995, class_weight='balanced')

    lr.fit(X_train, y_train)
    print(lr.coef_)
    y_pred_test = lr.predict(X_test)
    # 2W数据集：0.4998749061796347
    # 4W数据集：0.7124093493934135
    # 4W数据集CV5：0.8502412526760958
    print('ROC_AUC_SCORE:', roc_auc_score(y_test, y_pred_test))
    print('classification_report:\n', classification_report(y_test, y_pred_test))
    print('accuracy_score:', accuracy_score(y_test, y_pred_test))


if __name__ == '__main__':
    allData = load_data()
    X_train, X_test, y_train, y_test = split_test_data(allData)
    # 预处理
    # ss = StandardScaler()
    # X_train = ss.fit_transform(X_train)
    # X_test = ss.transform(X_test)
    LR(X_train, X_test, y_train, y_test)

