---
categories:
  - deepLearning
tags:
  - deepLearning
date: ‘2021-12-20'
mathjax: true
slug: dl-component
title: 深度学习基础——深度学习计算
---

前几篇介绍了常用概念以及最简单的线性回归和多层感知机，本篇介绍深度学习中的关键步骤组件，如模型构建、参数访问与初始化、设计自定义层和块、将模型读到磁盘以及利用GPU加速运算。

<!-- more -->

# 层和块

前几篇讨论的模型复杂度有限，最复杂的是含一个隐藏层的多层感知机模型，然而实际情况中，这种模型过于简单，表达效果不够好，因此需要研究较为复杂的模型。然而较为复杂的模型也是由简单模型组成，因此有必要研究简单模型如何组成复杂模型。事实证明可以通过”堆叠”简单模型实现，为了描述如何堆叠，引入块（block）的概念。块可以描述单层、多层组成的组件或者整个模型。使用块的一个好处是可以将一些块组合成更大的组件，这一过程通常是递归的，因此可以通过定义代码来生成任意复杂度的块，可以通过简介的代码实现复杂的神经网络。

## 自定义块

从程序实现的角度，块由类（class）实现。其子类定义一个将其输入转换为输出的前向传播函数，并存储必须的参数（如果需要的话）。为了计算梯度，块必须具有反向传播函数。所以，每个块必须具有的功能为：

1. 将输入数据作为前向传播函数的参数

2. 通过其前向传播函数生成输出

3. 计算输出关于输入的梯度，可通过反向传播函数访问

4. 存储和访问前向传播计算所需的参数

5. 根据需要初始化模型参数

此处实现一个自定义的类，包含一个多层感知机

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

前向传播函数以x为输入，计算带有激活函数的隐藏表示，并输出为规范化的输出值。注意关键细节：首先，_init_函数通过`super()._init_()`调用父类的`_init_`函数，省去重复编写的麻烦。除非实现了一个新的运算符，否则不必担心反向传播函数或参数初始化，系统将自动生成这些。

## 顺序块

如果块之间是顺序执行的，可以定义一个顺序类（默认有Sqauential类，此处自己实现），其中需要定义两个关键函数：

1. 将块逐个追加到列表中的函数

2. 前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”

具体代码实现为：

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

`_init_()`函数将每个模块逐个添加到有序字典`_modules`中调用前向函数时，每个块都按照他们被添加的顺序执行。现在使用`MySequential`类实现多层感知机。

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

## 在前向函数中执行代码

`sequential`类使模型构造变得简单，允许我们组合新的架构而无需定义自己的类。然而，不是所有的架构都是简单的顺序架构。当需要更强的灵活性时，需要定义自己的块。如，在前向函数中执行任意的数学运算。

目前为止，网络中所有操作都对网络的激活值及网络的参数起作用。然而，有时候可能希望合并既不是上一层结果也不是可更新参数项目，即常数参数（constant parameter）。可以通过一下代码实现。

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```python
net = FixedHiddenMLP()
net(X)
```

此模型中，有隐藏层的权重在实例化时初始化即为常量，因此不会被反向传播更新。然后将这个固定层的输出通过一个全连接层。

# 参数管理

在上节选择架构并设置超参数之后，就可以进行训练。训练的目的是找到损失函数最小化的模型参数值。训练完成后，需要保存这些参数来进行下一步的预测，也可能将参数提取出来，以便在其他地方进行复用。为了便于操作，本节仍以具有单隐藏层的多层感知机为例说明。

```python
net = FixedHiddenMLP()
net(X)
```

通过sequential类定义模型时，可以通过索引来访问模型的任一层，此时的模型类似于一个列表，每层的参数存储在属性中。

```python
print(net[2].state_dict())
```

可以输出该层包含哪些参数，以及每个参数的值。注意，参数名称唯一标识每个参数。

## 参数访问

### 访问目标参数

模型中的每个参数都表现为参数类的一个实例。要对参数进行任何操作，首先需要访问底层的数值。下面的代码从第二个全连接层开始提取偏置参数，提取后返回一个参数类实例，并进一步访问该参数的值。

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

参数是复合的对象，包含值、梯度和额外信息。 这就是我们需要显式参数值的原因。 除了值之外，我们还可以访问每个参数的梯度。 在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态。

### 访问所有参数

当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。 当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂， 因为我们需要递归整个树来提取每个子块的参数。

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

还可以通过如下方式进行访问

```python
net.state_dict()['2.bias'].data
```

### 从嵌套块访问参数

当块嵌套时，访问嵌套块中的参数类似于访问嵌套列表。首先定义一个生成块的函数，然后将这些块组合到更大的块中。

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

可以通过如下方式访问嵌套块中的参数

```python
rgnet[0][1][0].bias.data
```

## 参数初始化

一般而言深度学习框架提供默认随机初始化，也允许我们创建自定义初始化方法，满足我们通过其他规则实现初始化权重。

### 内置初始化

可以调用内置的初始化器，以下代码将所有权重参数初始化为标准差为0.01的高斯随机变量，并将偏置参数设置为0。

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

还可以将所有参数初始化为给定的常数，例如为1

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

此外，还可以针对不同块采用不同的初始化方法。例如，使用Xavier初始化第一个神经网络层，第三层初始化为常量42.

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

### 自定义初始化

如果深度学习框架没有提供我们需要的初始化方法，可以进行自定义。例如，可以通过如下分布为任意权重参数定义初始化方法。

$$
w \sim\left\{\begin{array}{ll}
U(5,10) & \text { 可能性 } \frac{1}{4} \\
0 & \text { 可能性 } \frac{1}{2} \\
U(-10,-5) & \text { 可能性 } \frac{1}{4}
\end{array}\right.
$$

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]


net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

## 参数绑定

有时希望在多层之间共享参数，可以定义一个共享层，然后使用其参数来设置另一层的参数。

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

上面代码中第三个和第五个神经网络层的参数是绑定的，不仅值相等，而且由相同的张量表示，如果改变其中一个参数，另一个参数也会改变。当参数绑定时，由于模型参数包含梯度，在反向传播时，绑定的两层的梯度会加在一起。

共享参数可以节省内存，此外在特定情况下还会有别的优势。例如：

- 对于图像识别中的CNN，共享参数使网络能够在图像中的任何地方而不是仅在某个区域中查找给定的功能。
- 对于RNN，它在序列的各个时间步之间共享参数，因此可以很好地推广到不同序列长度的示例。
- 对于自动编码器，编码器和解码器共享参数。 在具有线性激活的单层自动编码器中，共享权重会在权重矩阵的不同隐藏层之间强制正交。

# 延后初始化

上一节讨论了模型训练时参数如何初始化的问题，然而，仍然忽略了一些问题。

1. 定义了网络架构，却没有指定输入维度

2. 在添加 层时没有指定前一层的输出维度

3. 在初始化参数时，没有足够的信息来确定模型应该包含多少参数

然而，深度学习框架具有延后初始化（defers initialization）的特性，即知道数据第一次通过模型传递时，框架才会动态地推断每个层的大小。因此，在代码实现时无需知道维度即可设置参数，这种特定大大简化了定义和修改模型的复杂性。

# 自定义层

深度学习框架中提供了部分已经定义好的层，但是很可能遇到现有框架中不存在的层，因此需要自定义层。

## 不带参数的层

下面的代码定义了一个没有任何参数的层。`CenteredLayer`类要从其输入中减去均值。 要构建它，我们只需继承基础层类并实现前向传播功能

```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

向该层提供一些数据，验证是否能够按预期工作

```python
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

上述创建的层可以合并到更复杂的模型中。

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

## 带参数的层

自定义的层不仅可以带参数，也可以不带参数，参数可以通过模型训练调整。可以使用内置函数创建参数。以下代码实现自定义版本的全连接层。

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

使用该类访问其模型参数

```python
linear = MyLinear(5, 3)
linear.weight
```

使用自定义层执行前向传播运算

```python
linear(torch.rand(2, 5))
```

使用自定义层构建模型

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

# 读写文件

## 加载和保存张量

对于单个张量可以直接调用load和save函数去读写他们。这两个函数都要求提供一个名称，save要求将要保存的变量作为输入。

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

将存储在文件中的数据读回内存，也可以存储张量列表在读回内存

```python
x2 = torch.load('x-file')
x2
```

```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
```

写入或读取从字符串映射到张量的字典，可用于读取或写入模型中的所有权重。

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
```

## 加载和保存模型参数

上节介绍了保存单个权重向量。如果想保存整个模型以便于以后加载，单独保存每个向量会很麻烦。因此，深度学习框架提供了内置函数来保存和加载整个网络，以多层感知机为例。

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

将模型的参数存储在文件中

```python
torch.save(net.state_dict(), 'mlp.params')
```

如果要回复模型，可以直接读取文件中的参数，不需要初始化模型参数

```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

# GPU

本节介绍GPU计算常用的一些命令

`!nvidia-smi`——查看显卡信息

可以指定用于存储和计算的设备，如CPU或GPU。默认情况下，张量在内存中创建，然后使用CPU创建。

在PyTorch中，CPU和GPU可以用`torch.device('cpu')` 和`torch.device('cuda')`表示。 应该注意的是，`cpu`设备意味着所有物理CPU和内存， 这意味着PyTorch的计算将尝试使用所有CPU核心。 然而，`gpu`设备只代表一个卡和相应的显存。 如果有多个GPU，我们使用`torch.device(f'cuda:{i}')` 来表示第𝑖块GPU（𝑖从0开始）。 另外，`cuda:0`和`cuda`是等价的。

可以使用如下代码查看GPU的数量

```python
torch.cuda.device_count()
```

使用如下代码查询张量所在的设备

```python
x = torch.tensor([1, 2, 3])
x.device
```

如果需要对多项进行操作，则所有项必须在同一个设备上，否则框架不知道在哪里计算，在哪里存储结果。

## 在GPU上创建张量

在GPU上创建存储张量可以使用如下代码。以下代码首先在第一个GPU上创建张量X（需要注意X不能超过GPU显存限制），然后在第二个GPU上创建随机张量

```python
X = torch.ones(2, 3, device=try_gpu())
X
```

```python
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

## 复制

如果要计算`X+Y`，需要决定在哪里执行操作。因为两个张量不在同一个设备上，所以需要进行复制，可以将X传输到第二个GPU然后进行操作，如果直接进行求和会导致异常。

```python
Z = X.cuda(1)
print(X)
print(Z)
Y + Z
```

通过输出可以看出此时Z在第二个GPU上，复制成功，可以进行求和。

特别地，假设Z已经存在于第二个GPU，此时执行`Z.cuda(1)`会返回Z，而不是复制并分配新内存。

容易知道，在不同GPU设备之间传输数据比进行计算慢得多，所以，需要小心拷贝操作。此外，如果打印张量或者将其转换为Numpy时，如果数据不在内存，需要先复制到内存，会导致额外的开销。

## 神经网络在GPU上运行

类似于张量，神经网络模型也可以指定设备，以下代码可以将模型参数放在GPU上。

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

调用同GPU上的张量，在同GPU上进行计算

```python
net(X)
```
