## 公积金贷款逾期预测

最终选用的模型是第一层lightgbm+xgboost+RF，第二层SVC的stacking模型。  

模型效果如下（在仅使用已还清贷款数据和逾期贷款数据的情况下）：
# ![](https://ws4.sinaimg.cn/large/006tNbRwly1fxiep5z5sqj31go0j8mzt.jpg)

负类样本甚至比正类样本多，不需要过采样和欠采样方法。

