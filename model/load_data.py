# -*- coding: utf-8 -*
# 加载数据

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import gc
from time import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn import preprocessing, metrics
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_data():
    train = pd.read_csv('../input/all.csv')
    # test = pd.read_csv('../input/test.csv')
    # test = pd.read_csv('../input/test.csv')
    print(train.columns)
    print("未drop掉dkye != 0的数据时shape:", train.shape)
    train = train[train.DKYE == 0]
    const_cols = [c for c in train.columns if train[c].nunique(dropna=False) == 1]
    # print(const_cols)
    for i in train.columns:
        print("type of", i, train[i].dtype)

    print("drop掉dkye != 0的数据时shape:", train.shape)


    # 处理日期
    # train['YDFKRQ'] = train['YDFKRQ'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    # drop features
    # GRZHZT:个人账户状态
    # BIRTHDAY:生日
    # YDFKRQ:约定放款日期
    # JKHTQDRQ:借款合同签订日期
    # TIPTOP_DEGREE:学历（因为重复了）
    # ZHIWU:因为含有'A'
    # PROCESSSTATE
    # JOB_TITLE:职称（因为重复了）
    # MARRIAGE_STATE:婚姻状况（因为重复了）
    # OCCUPATIONID:职业（因为重复了）
    # SSRQ 新数据集不需要drop掉
    # 'YQQC', 'YQBJ', 'SSYQBJJE', 'SSYQFXJE', 'SSYQLXJE' 新数据集不需要drop掉
    train = train.drop(['GRZHZT', 'BIRTHDAY', 'YDFKRQ', 'JKHTQDRQ', 'TIPTOP_DEGREE', 'ZHIWU', 'PROCESSSTATE', 'JOB_TITLE', 'MARRIAGE_STATE', 'OCCUPATIONID', 'SSRQ'], axis=1)
    not_used_col = ['STATEID', 'FXZE', 'YQBJZE', 'YQLXZE', 'LJYQQS', 'YQQC', 'YQBJ', 'SSYQBJJE', 'SSYQFXJE', 'SSYQLXJE']
    for col in not_used_col:
        train = train.drop(col, axis=1)

    # 缺省值填充
    for i in train.columns:
        train[i] = train[i].fillna(0)

    # drop掉YQYS和暂时drop掉Category
    train = train.drop(['YQYS', 'Category'], axis=1)
    # train = train.drop(['YQYS', 'BinCategory'], axis=1)
    ''' 数值型特征
        JTYSR 家庭月收入
        PJYSR 各人平均月收入
        HTDKJE 合同贷款金额
        JKHTLL 借款合同利率(年%)
        MONTHRETURNAMOUNT 月还款额
        PUNISHRATE 罚息日利率(万分之)
        FWZJ 房屋总价
        GFSFK 购房首付款
        UNITPRICE 购房单价
        ALREADYPAYRATE 已付房款比例
        DKFFE 贷款发放额
        DKYE 贷款余额
        HSBJZE 回收本金总额
        HSLXZE 回收利息总额
        FXZE 罚息总额 ?
        TQGHBJZE 提前归还本金总额
        YQBJZE 逾期本金总额 ?
        YQLXZE 逾期利息总额 ?
        LJYQQS 累计逾期期数 ?
        DKQS 贷款期数
        LLFDBL 利率浮动比例 只有0和1
        YQQC 逾期期次/还款期次 ?
        YQBJ 结转逾期本金 ?
        SSYQBJJE 实收逾期本金金额 ?
        SSYQFXJE 实收逾期罚息金额 ?
        SSYQLXJE 实收逾期利息金额 ?
    '''
    numeric_col = ['JTYSR', 'PJYSR', 'HTDKJE', 'JKHTLL', 'MONTHRETURNAMOUNT', 'PUNISHRATE', 'FWZJ', 'GFSFK', 'UNITPRICE',
                   'ALREADYPAYRATE', 'DKFFE', 'DKYE', 'HSBJZE', 'HSLXZE', 'TQGHBJZE','DKQS', 'LLFDBL']

    ''' 类别型特征
        EMPLOYETYPE 缴存率类型
        HYZK 婚姻状况
        JOB 职务
        XINGBIE 性别
        XUELI 学历
        ZHICHEN 职称
        ZHIYE 职业
        max(DQQC) max(当期期次)
        PURPOSEID 贷款用途
    '''
    category_col = ['EMPLOYETYPE', 'HYZK', 'JOB', 'XINGBIE',
                'XUELI', 'ZHICHEN', 'ZHIYE', 'max(DQQC)', 'PURPOSEID']

    for col in category_col:
        train[col] = train[col].astype("category")

    for col in numeric_col:
        train[col] = train[col].astype(float)

    for col in category_col:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))

    print(train.columns)
    return train


#
def get_folds(df=None, n_splits=5):
    unique_vis = np.array(sorted(df['DKZH']))

    folds = GroupKFold(n_splits=n_splits)
    folds_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        folds_ids.append([
            ids[df['DKZH'].isin(unique_vis[trn_vis])],
            ids[df['DKZH'].isin(unique_vis[val_vis])]
        ])

    return folds_ids


def split_test_data(df=None):
    X, y = df.ix[:, 0:-1].values, df.ix[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1995)  # 随机选择20%作为测试集，剩余作为训练集

    return X_train, X_test, y_train, y_test

# 非线性降维TSNE
# 还可以用pca
def getTSNE(df=None, label=None):
    tsne = TSNE(n_components=2, init='pca', random_state=1995).fit_transform(df)
    plt.figure(figsize=(6, 6))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=label, marker='o')
    plt.colorbar()
    plt.show()


def getPCA(df=None, label=None):
    model = PCA(n_components=2)
    pca = model.fit_transform(df)
    plt.figure(figsize=(6, 6))
    plt.scatter(pca[:, 0], pca[:, 1], c=label, marker='o')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    allData = load_data()
    X_all = allData.ix[:, 0:-1]
    y_all = allData.ix[:, -1]
    print(X_all.shape, y_all.shape)
    X_train, X_test, y_train, y_test = split_test_data(allData)

    getPCA(X_test, y_test)

    # 总共40949条数据
    print("train集X大小为：", X_train.shape)
    print("test集X大小为：", X_test.shape)
    print("train集y大小为：", y_train.shape)
    print("test集y大小为：", y_test.shape)




