from torch import nn
import torch
import ray
import time
ray.init(address="auto")

def b0(x):
    conv1_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
    x = conv1_conv(x)
    conv1_relu = nn.ReLU(inplace=True)
    x = conv1_relu(x)
    maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    x = maxpool1(x)
    conv2_conv = nn.Conv2d(64, 64, kernel_size=1)
    x = conv2_conv(x)
    conv2_relu = nn.ReLU(inplace=True)
    x = conv2_relu(x)
    conv3_conv = nn.Conv2d(64, 192, kernel_size=3, padding=1)
    x = conv3_conv(x)
    conv3_relu = nn.ReLU(inplace=True)
    x = conv3_relu(x)
    maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    x = maxpool2(x)
    return x


@ray.remote
def b1_0(x):
    time.sleep(2)
    inception3a_branch1_conv = nn.Conv2d(192, 64, kernel_size=1)
    x = inception3a_branch1_conv(x)
    inception3a_branch1_relu = nn.ReLU(inplace=True)
    x = inception3a_branch1_relu(x)
    return x


@ray.remote
def b1_1(x):
    time.sleep(2)
    inception3a_branch2_0_conv = nn.Conv2d(192, 96, kernel_size=1)
    x = inception3a_branch2_0_conv(x)
    inception3a_branch2_0_relu = nn.ReLU(inplace=True)
    x = inception3a_branch2_0_relu(x)
    inception3a_branch2_1_conv = nn.Conv2d(96, 128, kernel_size=3, padding=1)
    x = inception3a_branch2_1_conv(x)
    inception3a_branch2_1_relu = nn.ReLU(inplace=True)
    x = inception3a_branch2_1_relu(x)
    return x


@ray.remote
def b1_2(x):
    time.sleep(2)
    inception3a_branch3_0_conv = nn.Conv2d(192, 16, kernel_size=1)
    x = inception3a_branch3_0_conv(x)
    inception3a_branch3_0_relu = nn.ReLU(inplace=True)
    x = inception3a_branch3_0_relu(x)
    inception3a_branch3_1_conv = nn.Conv2d(16, 32, kernel_size=5, padding=2)
    x = inception3a_branch3_1_conv(x)
    inception3a_branch3_1_relu = nn.ReLU(inplace=True)
    x = inception3a_branch3_1_relu(x)
    return x


@ray.remote
def b1_3(x):
    time.sleep(2)
    inception3a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    x = inception3a_branch4_0(x)
    inception3a_branch4_1_conv = nn.Conv2d(192, 32, kernel_size=1)
    x = inception3a_branch4_1_conv(x)
    inception3a_branch4_1_relu = nn.ReLU(inplace=True)
    x = inception3a_branch4_1_relu(x)
    return x

# @ray.remote
def b2(inception3a_branch1_relu, inception3a_branch2_1_relu, inception3a_branch3_1_relu, inception3a_branch4_1_relu):
    cat = torch.cat([ray.get(inception3a_branch1_relu),
                     ray.get(inception3a_branch2_1_relu),
                     ray.get(inception3a_branch3_1_relu),
                     ray.get(inception3a_branch4_1_relu)], 1)

    # cat = torch.cat([inception3a_branch1_relu,
    #                  inception3a_branch2_1_relu,
    #                  inception3a_branch3_1_relu,
    #                  inception3a_branch4_1_relu], 1)
    return cat

def forward(x):
    x = b0(x)
    _b1_0 = b1_0.remote(x)
    _b1_1 = b1_1.remote(x)
    _b1_2 = b1_2.remote(x)
    _b1_3 = b1_3.remote(x)
    x = b2(_b1_0, _b1_1, _b1_2, _b1_3)
    return x

# def forward(x):
#     x = b0(x)
#     _b1_0 = b1_0(x)
#     _b1_1 = b1_1(x)
#     _b1_2 = b1_2(x)
#     _b1_3 = b1_3(x)
#     x = b2(_b1_0, _b1_1, _b1_2, _b1_3)
#     return x



t1 = time.time()
# for i in range(100):
x = torch.randn(1, 1, 96, 96)
y = forward(x)
t2 = time.time() - t1
print(f"time： {t2}s")
print(y.shape)

