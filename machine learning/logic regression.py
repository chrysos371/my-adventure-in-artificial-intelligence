"""
逻辑回归案例：癌症分类预测
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #分割数据集为训练集和测试集
from sklearn.model_selection import GridSearchCV     #交叉验证和网格搜索
from sklearn.metrics import accuracy_score           #模型评估
from sklearn.preprocessing import StandardScaler     #数据标准化
from sklearn.metrics import confusion_matrix

def trainmodel():
    #获取数据
    data = pd.read_csv('breast-cancer-wisconsin.csv')
    data.info()
    #缺失值处理
    data = data.replace('?', np.nan)
    data = data.dropna()
    #提取特征和标签
    x = data.iloc[:,1:-1]
    print('x.head():\n',x.head())
    y = data['Class']
    print('y.head():\n',y.head())
    #分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 22)
    #数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #模型训练
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    #模型评估
    y_pred = estimator.predict(x_test)
    print('y_pred:',y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:',accuracy)
#混淆矩阵
def demo01():
    #手搓数据集:六个恶性，四个良性
    y_true = ['恶性','恶性','恶性','恶性','恶性','恶性','良性','良性','良性','良性']
    label = ['恶性','良性']
    dataframe_label = ['恶性（正例）','良性（反例）']
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    print('模型A:')
    print('-' * 13)
    y_pred1= ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']
    result = confusion_matrix(y_true, y_pred1,labels=label)
    print(pd.DataFrame(result, columns=dataframe_label,  index=dataframe_label))
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    print('模型B:')
    print('-' * 13)
    y_pred2= ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性']
    result = confusion_matrix(y_true, y_pred2, labels=label)
    print(pd.DataFrame(result,  columns=dataframe_label,  index=dataframe_label))


if __name__ == '__main__':
    demo01()

