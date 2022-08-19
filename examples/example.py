import torch
from torch import nn
from torch.nn.parameter import Parameter
import ray

ray.init(address="auto")


def seq1(x):
    p1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    x = p1(x)
    p2 = nn.ReLU(inplace=True)
    x = p2(x)
    p3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    x = p3(x)
    p4 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    x = p4(x)
    p5 = nn.ReLU(inplace=True)
    x = p5(x)
    p6 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    x = p6(x)
    p7 = nn.ReLU(inplace=True)
    x = p7(x)
    return x


@ray.remote
def branch1(x):
    p1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
    x = p1(x)
    p2 = nn.ReLU(inplace=True)
    x = p2(x)
    return x


@ray.remote
def branch2(x):
    p1 = nn.Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
    x = p1(x)
    p2 = nn.ReLU(inplace=True)
    x = p2(x)
    p3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    x = p3(x)
    p4 = nn.ReLU(inplace=True)
    x = p4(x)
    return x


@ray.remote
def branch3(x):
    p1 = nn.Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
    # p1.weight = Parameter(torch.randn(size=(16, 192, 1, 1)))
    # p1.bias = Parameter(torch.randn(size=(16,)))
    x = p1(x)
    p2 = nn.ReLU(inplace=True)
    x = p2(x)
    p3 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    x = p3(x)
    p4 = nn.ReLU(inplace=True)
    x = p4(x)
    return x


@ray.remote
def branch4(x):
    p1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    x = p1(x)
    p2 = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
    x = p2(x)
    p3 = nn.ReLU(inplace=True)
    x = p3(x)
    return x


def seq2(b1, b2, b3, b4):
    return torch.cat((ray.get(b1),
                      ray.get(b2),
                      ray.get(b3),
                      ray.get(b4)), dim=1)


# 调用示例
x = torch.rand(size=(1, 3, 96, 96))
x = seq1(x)
b1 = branch1.remote(x)
b2 = branch2.remote(x)
b3 = branch3.remote(x)
b4 = branch4.remote(x)


result = seq2(b1, b2, b3, b4)

print(result.shape)
print(result)

