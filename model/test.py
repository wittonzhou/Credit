# -*- coding: utf-8 -*
# 用网上找的测试数据集测试模型效果

import pandas as pd


def load_test_data():
    allData = pd.read_csv('../input/test/loan.csv')
    print(allData.columns)

    loan_status = allData['loan_status']
    allData = allData.drop(['loan_status'], axis=1)
    allData.insert(0, 'loan_status', loan_status)
    print(allData['loan_status'].unique())
    return allData


if __name__ == '__main__':
    allData = load_test_data()
