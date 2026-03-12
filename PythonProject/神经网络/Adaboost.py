"""
葡萄酒分类案例
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier     # 集成学习
from sklearn.metrics import accuracy_score


def demo01():
    #读取数据
    df = pd.read_csv('train.csv')
    #特征处理，去掉一类
    df = df[df['class label']!=1]
    #准备特征值和标签
    x = df[['Alcohol','Hue']].values
    y = df['class label']
    #类别转换
    y = LabelEncoder().fit_transform(y)
    #划分数据
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    #实例化单决策树和Adaboost
    mytree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1,random_state = 42)
    myada = AdaBoostClassifier(base_estimator= mytree,n_estimators = 500,learning_rate = 0.1)

    #训练和评估
    mytree.fit(X_train, y_train)
    myscore = mytree.score(X_test, y_test)
    print(myscore)

    myada.fit(X_train, y_train)
    myscore = myada.score(X_test, y_test)
    print(myscore)
#K聚类算法


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score    #一个评价聚类的方法
def demo02():
    #手搓数据集，每个样本两个特征四个
    x,y = make_blobs(n_samples=100,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=42)
    plt.figure()
    plt.scatter(x[:,0],x[:,1],marker='o')
    plt.show()

    y_pred = KMeans(n_clusters=2,random_state=42).fit_predict(x)
    plt.scatter(x[:,0],x[:,1],c=y_pred)
    plt.show()
    print(calinski_harabasz_score(x,y_pred))
if __name__ == '__main__':
    demo02()
