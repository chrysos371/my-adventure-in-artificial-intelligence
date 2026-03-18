"""
案例复现：初学者的拙劣尝试
基于《imagenet classification with Deep Convolutional Networks》提出的Alexnet结构，进行拙劣的论文复现练习，
仅旨在强化对CNN结构的理解以及循序渐进对计算机视觉领域的深耕

本案例由于条件受限：将原文的Imagenet数据集改为使用规模更小的CIFAR_10数据集，并基于数据集的改动对原始论文的一些参数进行适应性调整

"""

#导包
import torch
import os
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.cuda')
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

#数据处理
def data_processing():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomCrop(64,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transform)
    return train_dataset, test_dataset

"""
    数据处理这部分，由于论文本身是使用了Imagenet的数据集，尺寸为224*224，这里基于我们是初学者且受条件限制
    先将CIFAR-10由本身的32*32调整为64*64，保留了论文本身提出的“随机反转”，“随机裁剪”等处理方式
"""

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        #这一板块通过五层卷积，对特征进行深度提取
        self.features = nn.Sequential(
            #第一个卷积层
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #第二个卷积层
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #第三个卷积层
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            #第四个卷积层
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            #第五个卷积层
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #这一板块通过全连接层对提取出来的特征进行组合
        self.classifier = nn.Sequential(
            #这是Alexnet结构新引入的结构，用于正则化，防止过拟合
            nn.Dropout(p = 0.5),
            #通过0.5的概率随机使一些神经元“暂时失效”，抑制过拟合

            #第一层全连接
            nn.Linear(16384, 4096),
            nn.ReLU(inplace=True),

            #第二个全连接
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            #输出层
            nn.Linear(4096, 10),
        )
        #权重的初始化
        self._initialize_weights()#写在init方法里后就能自动初始化
    #前向传播
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
"""
网络结构设计这里，保留了原本的五层卷积+三层全连接的结构，保留了一个较为深层的网络结构
五层卷积可以实现对图片特征的深度提取，全连接层的实现对提取出来特征进行组合
这里由于CIFAR数据集的分辨率（64*64）较小，原论文的Imagenet的分辨率较大（224*224）
减小了原有的卷积核大小，
池化层的设计保留了论文“重叠池化”的设计，在压缩的同时保留了更多的特征
"""
"""
对权重的初始化：
卷积层和全连接层均进行了高斯分布的初始化
在这个例子中发现若依照原论文偏置初始化为1的话，会导致初始的损失较高，可能需要更多的epoch来训练
所以我将初始偏置设为1
"""

def train_model():

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    #这里用来储存每轮的损失和准确率，用于传输出去给可视化提供数据

    #传入数据集
    dataloader_train, dataloader_test = data_processing()
    #将模型传到GPU上，加速训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #实例化模型
    model = AlexNet()
    #上传模型
    model = model.to(device)
    print(f"模型部署在: {device}")
    summary(model, input_size=(3, 64, 64))
    #实例化损失函数为交叉熵
    criterion = nn.CrossEntropyLoss()
    #这里Alexnet结构加入了“动量”和“权重衰减”这两个新的系数，前者用于加速收敛，后者用于缓解过拟合
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
    #学习率调整器，
    scheduler = ReduceLROnPlateau(optimizer, mode='max',factor=0.1,patience=5)
    epochs = 40
    #构建容器
    dataloader_train = torch.utils.data.DataLoader(dataloader_train, batch_size=256, shuffle=True, num_workers=2)
    dataloader_test = torch.utils.data.DataLoader(dataloader_test, batch_size=256, shuffle=True, num_workers=2)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start = time.time()#计时起点

        # 训练阶段
        for x, y in dataloader_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)  # 累计总loss（按样本数加权）
            _, predicted = torch.max(output, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        # 计算本轮训练的平均loss和acc
        train_avg_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        train_loss_list.append(train_avg_loss)
        train_acc_list.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in dataloader_test:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(output, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_avg_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        val_loss_list.append(val_avg_loss)
        val_acc_list.append(val_acc)

        # 更新学习率（根据验证集acc）
        scheduler.step(val_acc)

        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {train_avg_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.2f}% | '
              f'Time: {time.time() - start:.2f}s')

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './best_model.pth')

    print(f'训练结束，最优验证集准确率: {best_acc:.2f}%')

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list
"""
训练过程中，设计为训练集，测试集同时训练
一方面是防止过拟合
一方面可以根据测试集表现成果对学习率进行调整
前期用大学习率快速降低loss
后期用小学习率防止由于大学习率跳过最优解
"""

def draw_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 5.1 绘制损失曲线
    ax1.plot(range(1, 41), train_loss_list, label='train loss', color='red', linewidth=2, marker='o')
    ax1.plot(range(1, 41), val_loss_list, label='test loss', color='blue', linewidth=2, marker='s')
    ax1.set_title('AlexNet train/test loss line', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)  # 加网格，更易读

    # 5.2 绘制准确率曲线
    ax2.plot(range(1, 41), train_acc_list, label='train acc', color='green', linewidth=2, marker='o')
    ax2.plot(range(1, 41), val_acc_list, label='test acc', color='orange', linewidth=2, marker='s')
    ax2.set_title('AlexNet train/test acc line', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('acc rate（%）', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 5.3 调整布局+保存图片
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig('./alexnet_training_curve.png', dpi=300, bbox_inches='tight')  # 保存高清图片
    plt.show()  # 显示图片

    print(f'可视化曲线已保存为: alexnet_training_curve.png')

if __name__ == '__main__':
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = train_model()
    draw_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
"""
若最后的准确率达到80%以上算一个优秀的模型，若能达到接近90%算一个相当优秀的模型
"""



















