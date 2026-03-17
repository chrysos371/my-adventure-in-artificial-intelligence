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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomCrop(224,padding=4),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
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
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),

            #第二个全连接
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            #输出层
            nn.Linear(4096, 10),
        )
        #权重的初始化
        self._initialize_weights()
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
                nn.init.constant_(m.bias, 1)
def train_model():
    dataloader_train, dataloader_test = data_processing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    criterion = nn.CrossEntropyLoss()
    #这里Alexnet结构加入了“动量”和“权重衰减”这两个新的系数，前者用于加速收敛，后者用于缓解过拟合
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=5e-4)
    epochs = 30
    dataloader_train = torch.utils.data.DataLoader(dataloader_train, batch_size=128, shuffle=True, num_workers=2)
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0
        start = time.time()
        for x,y in dataloader_train:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            avg_loss = total_loss / len(dataloader_train)
            acc = 100 * correct / total
            print(f'Epoch {epoch}, Loss {loss.item()}, avg_loss {avg_loss}Accuracy {acc}% time:{time.time()-start:.2f}s')
    torch.save(model.state_dict(),'./model.pth')
if __name__ == '__main__':
    train_model()




















