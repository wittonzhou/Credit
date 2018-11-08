# -*- coding: utf-8 -*
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from model.load_data import load_data, split_test_data


def RF(X_train, X_test, y_train, y_test, features_list):
    clf = RandomForestClassifier(random_state=1995)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    print('ROC_AUC_SCORE:', roc_auc_score(y_test, y_pred_test))
    print('classification_report:\n', classification_report(y_test, y_pred_test))
    print('accuracy_score:', accuracy_score(y_test, y_pred_test))
    print('feature importance:', clf.feature_importances_)

    feature_importance = clf.feature_importances_

    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(8, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.draw()
    plt.show()


if __name__ == '__main__':
    allData = load_data()
    X_train, X_test, y_train, y_test = split_test_data(allData)
    features_list = allData.columns.values[0:-1]
    RF(X_train, X_test, y_train, y_test, features_list)