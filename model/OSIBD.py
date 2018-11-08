# -*- coding: utf-8 -*
from sklearn.preprocessing import StandardScaler
from model.load_data import load_data, split_test_data
from sklearn import tree
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


'''
OSIBD
（1）准备多数类子集和少数类子集
（2）去噪声和边界实例来改善类不平衡
（3）少数类上过采样
（4）形成平衡的数据集，应用C4.5
'''


def get_maj_min_data(df=None):
    majroity = df[df.binaryCategory == 1]
    minroity = df[df.binaryCategory == -1]
    X_majroity = majroity.ix[:, 0:-1]
    X_minroity = minroity.ix[:, 0:-1]
    y_majority = majroity.ix[:, -1]
    y_minroity = minroity.ix[:, -1]
    return X_majroity, X_minroity, y_majority, y_minroity


def osibd(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(random_state=1995)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    # 未过采样 0.5130139073499438
    # 随机过采样 0.5265026969279092
    # SMOTE 0.5022771818200142
    # SMOTE-Borderline1 0.5138753354115113
    # ADASYN 0.5120060424465268
    print('ROC_AUC_SCORE:', roc_auc_score(y_test, y_pred_test))
    print('classification_report:\n', classification_report(y_test, y_pred_test))
    print('accuracy_score:', accuracy_score(y_test, y_pred_test))


def randomOverSampler(X_train, y_train):
    ros = RandomOverSampler(random_state=1995)
    X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
    majroity_size = y_resampled[y_resampled == 1]
    minroity_size = y_resampled[y_resampled == -1]
    print('majroity_size:', majroity_size.shape)
    print('minroity_size:', minroity_size.shape)
    return X_resampled, y_resampled


def smote(X_train, y_train):
    X_resampled, y_resampled = SMOTE(random_state=1995).fit_sample(X_train, y_train)
    majroity_size = y_resampled[y_resampled == 1]
    minroity_size = y_resampled[y_resampled == -1]
    print('majroity_size:', majroity_size.shape)
    print('minroity_size:', minroity_size.shape)
    return X_resampled, y_resampled


def smoteBorderline1(X_train, y_train):
    X_resampled, y_resampled = SMOTE(kind='borderline1', random_state=1995).fit_sample(X_train, y_train)
    majroity_size = y_resampled[y_resampled == 1]
    minroity_size = y_resampled[y_resampled == -1]
    print('majroity_size:', majroity_size.shape)
    print('minroity_size:', minroity_size.shape)
    return X_resampled, y_resampled


def adasyn(X_train, y_train):
    X_resampled, y_resampled = ADASYN(random_state=1995).fit_sample(X_train, y_train)
    majroity_size = y_resampled[y_resampled == 1]
    minroity_size = y_resampled[y_resampled == -1]
    print('过采样后majroity_size:', majroity_size.shape)
    print('过采样后minroity_size:', minroity_size.shape)
    return X_resampled, y_resampled


if __name__ == '__main__':
    allData = load_data()
    X_majroity, X_minroity, y_majority, y_minroity = get_maj_min_data(allData)
    print('未过采样时majroity_size:', y_majority.shape[0])
    print('未过采样时minroity_size:', y_minroity.shape[0])
    X_train, X_test, y_train, y_test = split_test_data(allData)

    # 过采样方法
    X_resampled, y_resampled = adasyn(X_train, y_train)

    # 预处理
    # ss = StandardScaler()
    # X_resampled = ss.fit_transform(X_resampled)
    # X_test = ss.transform(X_test)
    osibd(X_resampled, X_test, y_resampled, y_test)

