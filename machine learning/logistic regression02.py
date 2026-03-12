"""
逻辑回归案例：电信客户流失预测
"""
#导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split              #分割数据集为训练集和测试集
from sklearn.model_selection import GridSearchCV                  #交叉验证和网格搜索
from sklearn.metrics import accuracy_score,roc_auc_score          #模型评估
from sklearn.preprocessing import StandardScaler                  #数据标准化
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def dataprocessing():
    #导入数据
    churn = pd.read_csv('churn.csv')
    churn.info()
    print('churn.describe(): \n', churn.describe())
    print('churn:\n',churn)
    #对类别型数据进行处理，进行onehot编码
    churn = pd.get_dummies(churn)
    print('churn:\n',churn)
    churn.info()
    #删除了没必要的列
    churn.drop(['Churn_No','gender_Male'], axis=1, inplace=True)
    churn.info()
    #列标签重命名，并打印
    print('churn.columns: \n', churn.columns)
    churn.rename(columns={'Churn_Yes':'flag'}, inplace=True)
    print('churn.columns: \n', churn.columns)
    #查看标签分类情况
    value_counts = churn.flag.value_counts(1)
    print('value_counts:',value_counts)
#特征筛选
def fetureprocessing():
    churn = pd.read_csv('churn.csv')
    churn.info()
    # 对类别型数据进行处理，进行onehot编码
    churn = pd.get_dummies(churn)
    churn.info()
    # 删除了没必要的列
    churn.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn.info()
    # 列标签重命名，并打印
    churn.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 查看标签分类情况
    value_counts = churn.flag.value_counts(1)
    sns.countplot(data=churn,y='Contract_Month',hue='flag')
    plt.show()

#模型训练
def trainmodel():
    churn = pd.read_csv('churn.csv')
    churn.info()
    # 对类别型数据进行处理，进行onehot编码
    churn = pd.get_dummies(churn)
    churn.info()
    # 删除了没必要的列
    churn.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn.info()
    # 列标签重命名，并打印
    churn.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 查看标签分类情况
    value_counts = churn.flag.value_counts(1)
    #特征处理
    x = churn[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn['flag']
    #分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    #实例化模型，训练模型，模型预测
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    #模型评估
    my_accuracy = accuracy_score(y_test,y_pred)
    print('my_accuracy:',my_accuracy)
    my_score = estimator.score(x_test,y_test)
    print('my_score:',my_score)
    #计算AUC
    my_auc = roc_auc_score(y_test,y_pred)
    print('my_auc:',my_auc)

    result = classification_report(y_test,y_pred,target_names=['flag0','flag1'])
    print('result:\n',result)
if __name__ == '__main__':
    trainmodel()
