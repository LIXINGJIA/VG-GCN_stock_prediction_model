import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# layers中主要定义了图数据实现卷积操作的层，类似于CNN中的卷积层，只是一个层而已
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    # GraphConvolution作为一个类，首先需要定义其相关属性。主要定义了其输入特征in_feature、输出特征out_feature两个输入，
    # 以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 为了让每次训练产生的初始参数尽可能的相同，从而便于实验结果的复现，可以设置固定的随机数生成种子
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  # 标准偏差
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 此处主要定义的是本层的前向传播，通常采用的是 A ∗ X ∗ W A * X * WA∗X∗W的计算方法。由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # __repr__()方法是类的实例化对象用来做“自我介绍”的方法，默认情况下，它会返回当前对象的“类名+object at+内存地址”，
    # 而如果对该方法进行重写，可以为其制作自定义的自我描述信息
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'