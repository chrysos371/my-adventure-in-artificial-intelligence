"""
案例：加州房价预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Tools.i18n.pygettext import normalize
from sklearn.datasets import fetch_california_housing                #数据
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression       #正规方程的回归模型
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.preprocessing import StandardScaler        #特征处理
from sklearn.linear_model import SGDRegressor           #梯度下降的回归模型
from sklearn.metrics import mean_squared_error          #均方误差评估
from sklearn.linear_model import Ridge,RidgeCV

#正规方程
def trainmodel1():
    #获取数据
    boston = fetch_california_housing
    #数据集划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target,  random_state = 22)
    #特征工程标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #线性回归
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    #模型评估
    y_predict = estimator.predict(x_test)
    print('预测值为：\n', y_predict)
    print('模型的权重系数为：\n',estimator.coef_)
    print('模型的偏置为：\n',estimator.intercept_)
    #评价，均方误差
    error = mean_squared_error(y_test, y_predict)
    print('误差为：\n', error)

def trainmodel2():
    # 获取数据
    boston = fetch_california_housing
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 特征工程标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)

    x_test = transfer.transform(x_test)
    # 线性回归
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)
    # 模型评估
    y_predict = estimator.predict(x_test)
    print('预测值为：\n', y_predict)
    print('模型的权重系数为：\n', estimator.coef_)
    print('模型的偏置为：\n', estimator.intercept_)
    # 评价，均方误差
    error = mean_squared_error(y_test, y_predict)
    print('误差为：\n', error)
#欠拟合实例
def demo01():
    #准备数据下x,y，加上噪声
    np.random.seed(666)
    x = np.random.uniform(low=-3, high=3, size=100)
    y = 0.5 * x ** 2 + 2 + np.random.normal(0,1,size=100)
    #实例化线性回归模型
    estimator = LinearRegression()
    #训练模型
    X = x.reshape(-1, 1)
    estimator.fit(X, y)
    #模型预测
    y_predict = estimator.predict(X)
    #计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret:',myret)
    #画图
    plt.scatter(x, y)
    plt.plot(x,y_predict, color='red')
    plt.show()
#欠拟合补救
def demo02():
    # 准备数据下x,y，加上噪声
    np.random.seed(666)
    x = np.random.uniform(low=-3, high=3, size=100)
    y = 0.5 * x ** 2 + 2 + np.random.normal(0, 1, size=100)
    # 实例化线性回归模型
    estimator = LinearRegression()
    # 训练模型
    X = x.reshape(-1, 1)
    x2 = np.hstack([X,X**2])
    estimator.fit(x2, y)
    # 模型预测
    y_predict = estimator.predict(x2)
    # 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret:', myret)
    # 画图
    plt.scatter(x,y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()
#过拟合
def demo03():
    # 准备数据下x,y，加上噪声
    np.random.seed(666)
    x = np.random.uniform(low=-3, high=3, size=100)
    y = 0.5 * x ** 2 + 2 + np.random.normal(0, 1, size=100)
    # 实例化线性回归模型
    estimator = LinearRegression()
    # 训练模型
    X = x.reshape(-1, 1)
    x3 = np.hstack([X, X ** 2 , X**3, X**4, X**5, X**6, X**7, X**8,X**9, X**10])
    estimator.fit(x3, y)
    # 模型预测
    y_predict = estimator.predict(x3)
    # 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret:', myret)
    # 画图
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()
#对过拟合的模型进行lasso正则化
def demo04():
    # 准备数据下x,y，加上噪声
    np.random.seed(666)
    x = np.random.uniform(low=-3, high=3, size=100)
    y = 0.5 * x ** 2 + 2 + np.random.normal(0, 1, size=100)
    # 实例化线性回归模型
    estimator = Lasso(alpha=0.005,normalize = True)
    # 训练模型
    X = x.reshape(-1, 1)
    x3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    estimator.fit(x3, y)
    # 模型预测
    y_predict = estimator.predict(x3)
    # 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret:', myret)
    # 画图
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()
#对过拟合的模型进行Ridge正则化
def demo05():
    # 准备数据下x,y，加上噪声
    np.random.seed(666)
    x = np.random.uniform(low=-3, high=3, size=100)
    y = 0.5 * x ** 2 + 2 + np.random.normal(0, 1, size=100)
    # 实例化线性回归模型
    estimator = Ridge(alpha=0.005, normalize=True)
    # 训练模型
    X = x.reshape(-1, 1)
    x3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    estimator.fit(x3, y)
    # 模型预测
    y_predict = estimator.predict(x3)
    # 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret:', myret)
    # 画图
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()
if __name__ == '__main__':
    demo03()