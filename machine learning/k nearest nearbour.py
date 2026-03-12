"""
KNN算法介绍(K-Nearest Neighbors)，K近邻算法
原理
基于欧式距离(或者其他距离计算方式)计算测试集和每个训练集之间的距离，然后根据距离升序排列，找到最近的K个样本。
- 基于K个样本投票，票数多的就作为最终预测结果 → 分类问题
- 基于K个样本计算平均值，作为最终预测结果 → 回归问题

实现思路

1.分类问题
适用于：有特征，有标签，且标签是不连续的(离散的)
2.回归问题
适用于：有特征，有标签，且标签是连续的。

KNN算法，分类问题思路如下：

1.计算测试集和每个训练的样本之间的距离。
2.基于距离进行升序排列。
3.找到最近的K个样本。
4.K个样本进行投票。
5.票数多的结果，作为最终的预测结果。
"""
import numpy as np
#导包
from sklearn.neighbors import KNeighborsClassifier   #KNN算法类
from sklearn.datasets import load_iris               #载入鸢尾花数据集
from sklearn.model_selection import train_test_split #分割数据集为训练集和测试集
from sklearn.model_selection import GridSearchCV     #交叉验证和网格搜索
from sklearn.metrics import accuracy_score           #模型评估
from sklearn.preprocessing import StandardScaler     #数据标准化
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#准备数据集
def loadiris():
    #加载数据集
    mydataset = load_iris()
    #查看数据集信息
    print('\n查看数据集信息:\n', mydataset.data[:5])
    #查看目标值
    print('mydataset.target:\n', mydataset.target)
    #查看目标值名字
    print('mydataset.target_names:\n', mydataset.target_names)
    #查看特征名
    print('mydataset.feature_names:\n', mydataset.feature_names)
    #查看数据集描述
    print('\nmydataset.DESCR:\n', mydataset.DESCR)
    #数据文件路径
    print('mydataset.filename:\n', mydataset.filename)
#展示数据集
def showiris():
    #载入数据集，并显示数据名称
    mydataset = load_iris()
    print(mydataset.feature_names)
    #把数据转化为dataframe格式
    iris_d = pd.DataFrame(mydataset['data'], columns=mydataset.feature_names)
    iris_d['Species'] = mydataset.target
    print('\niris_d:\n',iris_d)
    col1 = 'sepal length (cm)'
    col2 = 'petal width (cm)'
    #sns,matplotlib显示
    sns.lmplot(x = col1, y = col2, data = iris_d,hue = 'Species',fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('iris_d')
    plt.show()
def traintest_split():
    #加载数据集
    mydataset = load_iris()
    #划分数据集
    X_train,X_test,Y_train,Y_test = train_test_split(mydataset['data'],mydataset['target'],test_size = 0.3,random_state = 22)
    print('数据总数量：',len(mydataset.data))
    print('训练集中的X特征值',len(X_train))
    print('训练集中的Y特征值',len(Y_train))
#创建模型对象
#模型训练
def trainmodel():
    mydataset = load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(mydataset['data'], mydataset['target'], test_size=0.3,
                                                        random_state=22)
    transfer = StandardScaler()#载入数据处理类
    X_train = transfer.fit_transform(X_train)#对训练集进行预处理
    X_test = transfer.transform(X_test)#对测试集进行预处理
    estimator = KNeighborsClassifier(n_neighbors=3)#载入knn算法类
    estimator.fit(X_train, Y_train)#将特征与标签进行匹配
    print('通过模型查看分类类别：',estimator.classes_)
    mydata = [[5.1,3.5,1.4,0.2],[4.6,3.1,1.5,0.2]]#自定义测试集
    mydata = transfer.transform(mydata)#预处理
    print('mydata:',mydata)
    mypred = estimator.predict(mydata)#输出预测的标签
    print('mypred:',mypred)
    mypred = estimator.predict_proba(mydata)#输出每个样本属于各类别的概率
    print('mypred:',mypred)
#交叉验证和网格搜索
def gridsearch():
    mydataset = load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(mydataset['data'], mydataset['target'], test_size=0.3,
                                                        random_state=22)
    transfer = StandardScaler()  # 载入数据处理类
    X_train = transfer.fit_transform(X_train)  # 对训练集进行预处理
    X_test = transfer.transform(X_test)  # 对测试集进行预处理
    estimator = KNeighborsClassifier(n_neighbors=3)  # 载入knn算法类
    param_grid = {'n_neighbors':[1,3,5,7]} #定义超参空间
    estimator = GridSearchCV(estimator = estimator,param_grid = param_grid,cv=5)
    estimator.fit(X_train, Y_train)
    print('estimator.best_score_:',estimator.best_score_)
    print('estimator.best_estimator_:',estimator.best_estimator_)
    print('estimator.best_params_:',estimator.best_params_)
    print('estimator.cv_results_:',estimator.cv_results_)
    myret = pd.DataFrame(estimator.cv_results_)
    myret.to_csv(path_or_buf='myret.csv')
    myscore = estimator.score(X_test, Y_test)
    print('myscore:', myscore)
#模型预测
if __name__ == '__main__':
    trainmodel()
    gridsearch()
