"""
案例：KNN算法识别手写数字，
介绍：
    每个图片都是由28 * 28个像素点构成，即：csv文件中的每一行都有784个像素点，表示图片每个像素点的颜色
    最终构成图像
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

def show_digit(idx):
    #读取数据集
    data = pd.read_csv('手写数字识别.csv')
    #判断是否越界，若没有就正常获取数据
    if idx < 0 or idx >= len(data) - 1:
        print('越界')
        return
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    #查看用户传入的索引是几
    print(f'该图片对应的数字是：{y.iloc[idx]}')
    #查看下 x 的形状
    x = x.iloc[idx].values.reshape(28, 28)
    #具体的绘制灰度图的动作
    plt.imshow(x, cmap='gray')
    plt.show()
def train_model():
    #加载手写数据集
    data = pd.read_csv('手写数字识别.csv')
    #数字预处理，归一化
    x = data.iloc[:, 1:] / 255
    y = data.iloc[:, 0]
    #分割数据集
    split_data = train_test_split(x, y, test_size=0.2, stratify=y,random_state=0)
    x_train, x_test, y_train, y_test = split_data
    #模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)
    #模型评估
    acc = estimator.score(x_test, y_test)
    print('测试集准确率：%.2f' % acc)
    #模型保存
    joblib.dump(estimator, 'knn_model.pkl')
def test_model():
    #读取图片数据
    img = plt.imread('demo.png')
    plt.imshow(img)
    #加载模型
    knn = joblib.load('knn_model.pkl')

    #预测图片
    y_pred = pd.DataFrame(img.reshape(1, -1))
    x = knn.predict(y_pred)
    print('您绘制的图片是',x)

if __name__ == '__main__':
    show_digit(23)
    train_model()
    test_model()