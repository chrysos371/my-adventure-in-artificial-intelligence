import torch
import numpy as np

def demo01():
    #创建零维张量
    data = torch.tensor(10)
    print(data)
    #结合numpy创建二维张量
    data = np.random.randn(2, 3)
    data = torch.tensor(data)
    print(data)
    #用列表创建二维张量
    data = [[10,20,30],[40,50,60]]
    data = torch.tensor(data)
    print(data)
    # 1. 创建2行3列, dtype 为 int32 的张量
    data = torch.IntTensor(2, 3)
    print(data)
    # 2. 注意: 如果传递的元素类型不正确, 则会进行类型转换
    data = torch.IntTensor([2.5, 3.3])
    print(data)
    # 3. 其他的类型
    data = torch.ShortTensor()  # int16
    data = torch.LongTensor()  # int64
    data = torch.FloatTensor()  # float32
    data = torch.DoubleTensor()  # float64
def demo02():
    #创建线性随机张量

    #按照步长生成元素
    data = torch.arange(0,10,2)
    print(data)
    #在指定区域根据元素个数生成
    data = torch.linspace(0,9,10)
    print(data)

    #种子的运用
    data = torch.arange(0,10,2)
    print(data)
    #查看随机数种子
    print('随机数种子：',torch.initial_seed())
    #随机数种子设置
    torch.manual_seed(100)
    data = torch.randn(2, 3)
    print(data)
    print('随机数种子：',torch.initial_seed())

    #创建全零张量
    data = torch.zeros(2,3)
    print(data)
    data = torch.zeros_like(data)
    print(data)
    #全一张量同理

    #创建全指定值张量
    data = torch.full((2,3),10)
    print(data)
    data = torch.full_like(data,10)
    print(data)

    #张量类型转化
    data = data.type(torch.DoubleTensor)
    print(data.dtype)
    #或者
    data = data.double()
def demo03():
    #张量和numpy数组可以相互转换
    data_tensor = torch.tensor([1,2,3,4,5,6])
    data_numpy = data_tensor.numpy() #转换
    #但是data_tensor和data_numpy共享内存，改一个的话另一个也会改
    #上面demo01里的方式则不会导致共享内存

    #对于只包含一个元素的张量，用item函数可以将其从中提取出来
    data = torch.tensor([30,])
    print(data.item())
    data = torch.tensor(30)
    print(data.item())
    #torch中的加减乘除
    #add()加
    #sub()减
    #mul()乘
    #div()除
    #若带有下划线会修改原数据，即自加/减/乘/除

    #点乘，即对应元素之间相乘得到新的张量
    data1 = torch.tensor([[1,2],[3,4]])
    data2 = torch.tensor([[5,6],[7,8]])
    data = torch.mul(data1,data2)#第一种方式
    data = data1 * data2         #第二种方式
    #矩阵乘法
    data = data1 @ data2                #第一种方式
    data = torch.matmul(data1,data2)    #第二种方式

    #计算均值
    print(data.mean())
    print(data.mean(dim=0))         #按列计算均值
    print(data.mean(dim=1))         #按行计算均值
    #计算总和
    print(data.sum())
    print(data.sum(dim=0))          #按列计算总和
    print(data.sum(dim=1))          #按行计算总和
    #计算平方
    print(torch.pow(data,2))
    #计算平方根
    print(data.sqrt())
    print(data.exp())               #自然对数e的data次方
    print(data.log())               #相当于ln
    print(data.log2())              #log以2为底
def demo04():
    #索引用法
    data = torch.randint(0,10,[4,5])
    print(data[0])                  #结果为第一行数据
    print(data[:,0])                #结果为第一列数据
    print(data[[0,1],[1,2]])        #结果为（0，1）（1，2）两个位置的元素
    print(data[[[0],[1]],[1,2]])    #结果为0，1行的一二元素，一共四个元素
    print(data[:3,:2])              #前三行的前两列元素
    print(data[2:,:2])              #第二行到最后的前两行元素
    print(data[data[:,2]>5])        #第三列大于五的行数据
    print(data[:, data[1] > 5])     # 第二行大于5的列数据
    #更高维的张量同理
def demo05():
    data1 = torch.randint(0,10,[1,2,3])
    data2 = torch.randint(0,10,[1,2,3])
    #按零维度拼接
    new_data = torch.cat([data1,data2],dim = 0)
    print(new_data)
    print(new_data.shape)
    #二维三维拼接同理

    new_data = torch.stack([data1,data2],dim = 0)
    #同样是按照零维拼接，但会增加一个维度

    #自动微分模块
    x = torch.tensor(10,requires_grad=True,dtype = torch.float32)#此时X就是一个标量
    y = 2*x**2
    y.sum().backward()
    print(x.grad)
    #若X不是一个标量，则要对Y进行转换，即y.sum()

    #利用梯度下降法求最优解
    x = torch.tensor(10,requires_grad=True,dtype = torch.float32)
    y = x**2 + 20
    # f-string 版本（替换 % 格式化）
    print(f'开始 权重x初始值:{x:.6f} (0.01 * x.grad):无 y:{y:.6f}')
    for i in range(1,1001):
        y = x ** 2 + 20
        if x.grad is not None:
            x.grad.zero_()
        y.backward()
        x.data = x.data - 0.01 * x.grad
        print('次数:%d 权重x: %.6f, (0.01 * x.grad):%.6f y:%.6f\n' % (i, x, 0.01 * x.grad, y))
    print(y)
    #不能将自动微分的张量转化为numpy数组，可以用detach方法，相当于复制了一份

if __name__ == '__main__':
    demo05()

