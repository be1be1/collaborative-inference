from torch import nn
from torch.nn import functional as F
import torch





#inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


#辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


import ray
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # #
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        #
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        #
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        # if init_weights:
        #     self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def b0(self, x):
        conv1_conv = self.conv1.conv(x);  x = None
        conv1_relu = self.conv1.relu(conv1_conv);  conv1_conv = None
        maxpool1 = self.maxpool1(conv1_relu);  conv1_relu = None
        conv2_conv = self.conv2.conv(maxpool1);  maxpool1 = None
        conv2_relu = self.conv2.relu(conv2_conv);  conv2_conv = None
        conv3_conv = self.conv3.conv(conv2_relu);  conv2_relu = None
        conv3_relu = self.conv3.relu(conv3_conv);  conv3_conv = None
        maxpool2 = self.maxpool2(conv3_relu);  conv3_relu = None
        return maxpool2

    @ray.remote(scheduling_strategy="SPREAD")
    def b1_0(self, maxpool2):
        inception3a_branch1_conv = self.inception3a.branch1.conv(maxpool2)
        inception3a_branch1_relu = self.inception3a.branch1.relu(inception3a_branch1_conv);  inception3a_branch1_conv = None
        return inception3a_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b1_1(self, maxpool2):
        inception3a_branch2_0_conv = getattr(self.inception3a.branch2, "0").conv(maxpool2)
        inception3a_branch2_0_relu = getattr(self.inception3a.branch2, "0").relu(inception3a_branch2_0_conv);  inception3a_branch2_0_conv = None
        inception3a_branch2_1_conv = getattr(self.inception3a.branch2, "1").conv(inception3a_branch2_0_relu);  inception3a_branch2_0_relu = None
        inception3a_branch2_1_relu = getattr(self.inception3a.branch2, "1").relu(inception3a_branch2_1_conv);  inception3a_branch2_1_conv = None
        return inception3a_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b1_2(self, maxpool2):
        inception3a_branch3_0_conv = getattr(self.inception3a.branch3, "0").conv(maxpool2)
        inception3a_branch3_0_relu = getattr(self.inception3a.branch3, "0").relu(inception3a_branch3_0_conv);  inception3a_branch3_0_conv = None
        inception3a_branch3_1_conv = getattr(self.inception3a.branch3, "1").conv(inception3a_branch3_0_relu);  inception3a_branch3_0_relu = None
        inception3a_branch3_1_relu = getattr(self.inception3a.branch3, "1").relu(inception3a_branch3_1_conv);  inception3a_branch3_1_conv = None
        return inception3a_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b1_3(self, maxpool2):
        inception3a_branch4_0 = getattr(self.inception3a.branch4, "0")(maxpool2);  maxpool2 = None
        inception3a_branch4_1_conv = getattr(self.inception3a.branch4, "1").conv(inception3a_branch4_0);  inception3a_branch4_0 = None
        inception3a_branch4_1_relu = getattr(self.inception3a.branch4, "1").relu(inception3a_branch4_1_conv);  inception3a_branch4_1_conv = None
        return inception3a_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b2(self, inception3a_branch1_relu, inception3a_branch2_1_relu, inception3a_branch3_1_relu, inception3a_branch4_1_relu):
        cat = torch.cat([inception3a_branch1_relu, inception3a_branch2_1_relu, inception3a_branch3_1_relu, inception3a_branch4_1_relu], 1);  inception3a_branch1_relu = inception3a_branch2_1_relu = inception3a_branch3_1_relu = inception3a_branch4_1_relu = None
        return cat

    @ray.remote(scheduling_strategy="SPREAD")
    def b3_0(self, cat):
        inception3b_branch1_conv = self.inception3b.branch1.conv(cat)
        inception3b_branch1_relu = self.inception3b.branch1.relu(inception3b_branch1_conv);  inception3b_branch1_conv = None
        return inception3b_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b3_1(self, cat):
        inception3b_branch2_0_conv = getattr(self.inception3b.branch2, "0").conv(cat)
        inception3b_branch2_0_relu = getattr(self.inception3b.branch2, "0").relu(inception3b_branch2_0_conv);  inception3b_branch2_0_conv = None
        inception3b_branch2_1_conv = getattr(self.inception3b.branch2, "1").conv(inception3b_branch2_0_relu);  inception3b_branch2_0_relu = None
        inception3b_branch2_1_relu = getattr(self.inception3b.branch2, "1").relu(inception3b_branch2_1_conv);  inception3b_branch2_1_conv = None
        return inception3b_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b3_2(self, cat):
        inception3b_branch3_0_conv = getattr(self.inception3b.branch3, "0").conv(cat)
        inception3b_branch3_0_relu = getattr(self.inception3b.branch3, "0").relu(inception3b_branch3_0_conv);  inception3b_branch3_0_conv = None
        inception3b_branch3_1_conv = getattr(self.inception3b.branch3, "1").conv(inception3b_branch3_0_relu);  inception3b_branch3_0_relu = None
        inception3b_branch3_1_relu = getattr(self.inception3b.branch3, "1").relu(inception3b_branch3_1_conv);  inception3b_branch3_1_conv = None
        return inception3b_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b3_3(self, cat):
        inception3b_branch4_0 = getattr(self.inception3b.branch4, "0")(cat);  cat = None
        inception3b_branch4_1_conv = getattr(self.inception3b.branch4, "1").conv(inception3b_branch4_0);  inception3b_branch4_0 = None
        inception3b_branch4_1_relu = getattr(self.inception3b.branch4, "1").relu(inception3b_branch4_1_conv);  inception3b_branch4_1_conv = None
        return inception3b_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b4(self, inception3b_branch1_relu, inception3b_branch2_1_relu, inception3b_branch3_1_relu, inception3b_branch4_1_relu):
        cat_1 = torch.cat([inception3b_branch1_relu, inception3b_branch2_1_relu, inception3b_branch3_1_relu, inception3b_branch4_1_relu], 1);  inception3b_branch1_relu = inception3b_branch2_1_relu = inception3b_branch3_1_relu = inception3b_branch4_1_relu = None
        maxpool3 = self.maxpool3(cat_1);  cat_1 = None
        return maxpool3

    @ray.remote(scheduling_strategy="SPREAD")
    def b5_0(self, maxpool3):
        inception4a_branch1_conv = self.inception4a.branch1.conv(maxpool3)
        inception4a_branch1_relu = self.inception4a.branch1.relu(inception4a_branch1_conv);  inception4a_branch1_conv = None
        return inception4a_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b5_1(self, maxpool3):
        inception4a_branch2_0_conv = getattr(self.inception4a.branch2, "0").conv(maxpool3)
        inception4a_branch2_0_relu = getattr(self.inception4a.branch2, "0").relu(inception4a_branch2_0_conv);  inception4a_branch2_0_conv = None
        inception4a_branch2_1_conv = getattr(self.inception4a.branch2, "1").conv(inception4a_branch2_0_relu);  inception4a_branch2_0_relu = None
        inception4a_branch2_1_relu = getattr(self.inception4a.branch2, "1").relu(inception4a_branch2_1_conv);  inception4a_branch2_1_conv = None
        return inception4a_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b5_2(self, maxpool3):
        inception4a_branch3_0_conv = getattr(self.inception4a.branch3, "0").conv(maxpool3)
        inception4a_branch3_0_relu = getattr(self.inception4a.branch3, "0").relu(inception4a_branch3_0_conv);  inception4a_branch3_0_conv = None
        inception4a_branch3_1_conv = getattr(self.inception4a.branch3, "1").conv(inception4a_branch3_0_relu);  inception4a_branch3_0_relu = None
        inception4a_branch3_1_relu = getattr(self.inception4a.branch3, "1").relu(inception4a_branch3_1_conv);  inception4a_branch3_1_conv = None
        return inception4a_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b5_3(self, maxpool3):
        inception4a_branch4_0 = getattr(self.inception4a.branch4, "0")(maxpool3);  maxpool3 = None
        inception4a_branch4_1_conv = getattr(self.inception4a.branch4, "1").conv(inception4a_branch4_0);  inception4a_branch4_0 = None
        inception4a_branch4_1_relu = getattr(self.inception4a.branch4, "1").relu(inception4a_branch4_1_conv);  inception4a_branch4_1_conv = None
        return inception4a_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b6(self, inception4a_branch1_relu, inception4a_branch2_1_relu, inception4a_branch3_1_relu, inception4a_branch4_1_relu):
        cat_2 = torch.cat([inception4a_branch1_relu, inception4a_branch2_1_relu, inception4a_branch3_1_relu, inception4a_branch4_1_relu], 1);  inception4a_branch1_relu = inception4a_branch2_1_relu = inception4a_branch3_1_relu = inception4a_branch4_1_relu = None
        return cat_2

    @ray.remote(scheduling_strategy="SPREAD")
    def b7_0(self, cat_2):
        inception4b_branch1_conv = self.inception4b.branch1.conv(cat_2)
        inception4b_branch1_relu = self.inception4b.branch1.relu(inception4b_branch1_conv);  inception4b_branch1_conv = None
        return inception4b_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b7_1(self, cat_2):
        inception4b_branch2_0_conv = getattr(self.inception4b.branch2, "0").conv(cat_2)
        inception4b_branch2_0_relu = getattr(self.inception4b.branch2, "0").relu(inception4b_branch2_0_conv);  inception4b_branch2_0_conv = None
        inception4b_branch2_1_conv = getattr(self.inception4b.branch2, "1").conv(inception4b_branch2_0_relu);  inception4b_branch2_0_relu = None
        inception4b_branch2_1_relu = getattr(self.inception4b.branch2, "1").relu(inception4b_branch2_1_conv);  inception4b_branch2_1_conv = None
        return inception4b_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b7_2(self, cat_2):
        inception4b_branch3_0_conv = getattr(self.inception4b.branch3, "0").conv(cat_2)
        inception4b_branch3_0_relu = getattr(self.inception4b.branch3, "0").relu(inception4b_branch3_0_conv);  inception4b_branch3_0_conv = None
        inception4b_branch3_1_conv = getattr(self.inception4b.branch3, "1").conv(inception4b_branch3_0_relu);  inception4b_branch3_0_relu = None
        inception4b_branch3_1_relu = getattr(self.inception4b.branch3, "1").relu(inception4b_branch3_1_conv);  inception4b_branch3_1_conv = None
        return inception4b_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b7_3(self, cat_2):
        inception4b_branch4_0 = getattr(self.inception4b.branch4, "0")(cat_2);  cat_2 = None
        inception4b_branch4_1_conv = getattr(self.inception4b.branch4, "1").conv(inception4b_branch4_0);  inception4b_branch4_0 = None
        inception4b_branch4_1_relu = getattr(self.inception4b.branch4, "1").relu(inception4b_branch4_1_conv);  inception4b_branch4_1_conv = None
        return inception4b_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b8(self, inception4b_branch1_relu, inception4b_branch2_1_relu, inception4b_branch3_1_relu, inception4b_branch4_1_relu):
        cat_3 = torch.cat([inception4b_branch1_relu, inception4b_branch2_1_relu, inception4b_branch3_1_relu, inception4b_branch4_1_relu], 1);  inception4b_branch1_relu = inception4b_branch2_1_relu = inception4b_branch3_1_relu = inception4b_branch4_1_relu = None
        return cat_3

    @ray.remote(scheduling_strategy="SPREAD")
    def b9_0(self, cat_3):
        inception4c_branch1_conv = self.inception4c.branch1.conv(cat_3)
        inception4c_branch1_relu = self.inception4c.branch1.relu(inception4c_branch1_conv);  inception4c_branch1_conv = None
        return inception4c_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b9_1(self, cat_3):
        inception4c_branch2_0_conv = getattr(self.inception4c.branch2, "0").conv(cat_3)
        inception4c_branch2_0_relu = getattr(self.inception4c.branch2, "0").relu(inception4c_branch2_0_conv);  inception4c_branch2_0_conv = None
        inception4c_branch2_1_conv = getattr(self.inception4c.branch2, "1").conv(inception4c_branch2_0_relu);  inception4c_branch2_0_relu = None
        inception4c_branch2_1_relu = getattr(self.inception4c.branch2, "1").relu(inception4c_branch2_1_conv);  inception4c_branch2_1_conv = None
        return inception4c_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b9_2(self, cat_3):
        inception4c_branch3_0_conv = getattr(self.inception4c.branch3, "0").conv(cat_3)
        inception4c_branch3_0_relu = getattr(self.inception4c.branch3, "0").relu(inception4c_branch3_0_conv);  inception4c_branch3_0_conv = None
        inception4c_branch3_1_conv = getattr(self.inception4c.branch3, "1").conv(inception4c_branch3_0_relu);  inception4c_branch3_0_relu = None
        inception4c_branch3_1_relu = getattr(self.inception4c.branch3, "1").relu(inception4c_branch3_1_conv);  inception4c_branch3_1_conv = None
        return inception4c_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b9_3(self, cat_3):
        inception4c_branch4_0 = getattr(self.inception4c.branch4, "0")(cat_3);  cat_3 = None
        inception4c_branch4_1_conv = getattr(self.inception4c.branch4, "1").conv(inception4c_branch4_0);  inception4c_branch4_0 = None
        inception4c_branch4_1_relu = getattr(self.inception4c.branch4, "1").relu(inception4c_branch4_1_conv);  inception4c_branch4_1_conv = None
        return inception4c_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b10(self, inception4c_branch1_relu, inception4c_branch2_1_relu, inception4c_branch3_1_relu, inception4c_branch4_1_relu):
        cat_4 = torch.cat([inception4c_branch1_relu, inception4c_branch2_1_relu, inception4c_branch3_1_relu, inception4c_branch4_1_relu], 1);  inception4c_branch1_relu = inception4c_branch2_1_relu = inception4c_branch3_1_relu = inception4c_branch4_1_relu = None
        return cat_4

    @ray.remote(scheduling_strategy="SPREAD")
    def b11_0(self, cat_4):
        inception4d_branch1_conv = self.inception4d.branch1.conv(cat_4)
        inception4d_branch1_relu = self.inception4d.branch1.relu(inception4d_branch1_conv);  inception4d_branch1_conv = None
        return inception4d_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b11_1(self, cat_4):
        inception4d_branch2_0_conv = getattr(self.inception4d.branch2, "0").conv(cat_4)
        inception4d_branch2_0_relu = getattr(self.inception4d.branch2, "0").relu(inception4d_branch2_0_conv);  inception4d_branch2_0_conv = None
        inception4d_branch2_1_conv = getattr(self.inception4d.branch2, "1").conv(inception4d_branch2_0_relu);  inception4d_branch2_0_relu = None
        inception4d_branch2_1_relu = getattr(self.inception4d.branch2, "1").relu(inception4d_branch2_1_conv);  inception4d_branch2_1_conv = None
        return inception4d_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b11_2(self, cat_4):
        inception4d_branch3_0_conv = getattr(self.inception4d.branch3, "0").conv(cat_4)
        inception4d_branch3_0_relu = getattr(self.inception4d.branch3, "0").relu(inception4d_branch3_0_conv);  inception4d_branch3_0_conv = None
        inception4d_branch3_1_conv = getattr(self.inception4d.branch3, "1").conv(inception4d_branch3_0_relu);  inception4d_branch3_0_relu = None
        inception4d_branch3_1_relu = getattr(self.inception4d.branch3, "1").relu(inception4d_branch3_1_conv);  inception4d_branch3_1_conv = None
        return inception4d_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b11_3(self, cat_4):
        inception4d_branch4_0 = getattr(self.inception4d.branch4, "0")(cat_4);  cat_4 = None
        inception4d_branch4_1_conv = getattr(self.inception4d.branch4, "1").conv(inception4d_branch4_0);  inception4d_branch4_0 = None
        inception4d_branch4_1_relu = getattr(self.inception4d.branch4, "1").relu(inception4d_branch4_1_conv);  inception4d_branch4_1_conv = None
        return inception4d_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b12(self, inception4d_branch1_relu, inception4d_branch2_1_relu, inception4d_branch3_1_relu, inception4d_branch4_1_relu):
        cat_5 = torch.cat([inception4d_branch1_relu, inception4d_branch2_1_relu, inception4d_branch3_1_relu, inception4d_branch4_1_relu], 1);  inception4d_branch1_relu = inception4d_branch2_1_relu = inception4d_branch3_1_relu = inception4d_branch4_1_relu = None
        return cat_5

    @ray.remote(scheduling_strategy="SPREAD")
    def b13_0(self, cat_5):
        inception4e_branch1_conv = self.inception4e.branch1.conv(cat_5)
        inception4e_branch1_relu = self.inception4e.branch1.relu(inception4e_branch1_conv);  inception4e_branch1_conv = None
        return inception4e_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b13_1(self, cat_5):
        inception4e_branch2_0_conv = getattr(self.inception4e.branch2, "0").conv(cat_5)
        inception4e_branch2_0_relu = getattr(self.inception4e.branch2, "0").relu(inception4e_branch2_0_conv);  inception4e_branch2_0_conv = None
        inception4e_branch2_1_conv = getattr(self.inception4e.branch2, "1").conv(inception4e_branch2_0_relu);  inception4e_branch2_0_relu = None
        inception4e_branch2_1_relu = getattr(self.inception4e.branch2, "1").relu(inception4e_branch2_1_conv);  inception4e_branch2_1_conv = None
        return inception4e_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b13_2(self, cat_5):
        inception4e_branch3_0_conv = getattr(self.inception4e.branch3, "0").conv(cat_5)
        inception4e_branch3_0_relu = getattr(self.inception4e.branch3, "0").relu(inception4e_branch3_0_conv);  inception4e_branch3_0_conv = None
        inception4e_branch3_1_conv = getattr(self.inception4e.branch3, "1").conv(inception4e_branch3_0_relu);  inception4e_branch3_0_relu = None
        inception4e_branch3_1_relu = getattr(self.inception4e.branch3, "1").relu(inception4e_branch3_1_conv);  inception4e_branch3_1_conv = None
        return inception4e_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b13_3(self, cat_5):
        inception4e_branch4_0 = getattr(self.inception4e.branch4, "0")(cat_5);  cat_5 = None
        inception4e_branch4_1_conv = getattr(self.inception4e.branch4, "1").conv(inception4e_branch4_0);  inception4e_branch4_0 = None
        inception4e_branch4_1_relu = getattr(self.inception4e.branch4, "1").relu(inception4e_branch4_1_conv);  inception4e_branch4_1_conv = None
        return inception4e_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b14(self, inception4e_branch1_relu, inception4e_branch2_1_relu, inception4e_branch3_1_relu, inception4e_branch4_1_relu):
        cat_6 = torch.cat([inception4e_branch1_relu, inception4e_branch2_1_relu, inception4e_branch3_1_relu, inception4e_branch4_1_relu], 1);  inception4e_branch1_relu = inception4e_branch2_1_relu = inception4e_branch3_1_relu = inception4e_branch4_1_relu = None
        maxpool4 = self.maxpool4(cat_6);  cat_6 = None
        return maxpool4

    @ray.remote(scheduling_strategy="SPREAD")
    def b15_0(self, maxpool4):
        inception5a_branch1_conv = self.inception5a.branch1.conv(maxpool4)
        inception5a_branch1_relu = self.inception5a.branch1.relu(inception5a_branch1_conv);  inception5a_branch1_conv = None
        return inception5a_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b15_1(self, maxpool4):
        inception5a_branch2_0_conv = getattr(self.inception5a.branch2, "0").conv(maxpool4)
        inception5a_branch2_0_relu = getattr(self.inception5a.branch2, "0").relu(inception5a_branch2_0_conv);  inception5a_branch2_0_conv = None
        inception5a_branch2_1_conv = getattr(self.inception5a.branch2, "1").conv(inception5a_branch2_0_relu);  inception5a_branch2_0_relu = None
        inception5a_branch2_1_relu = getattr(self.inception5a.branch2, "1").relu(inception5a_branch2_1_conv);  inception5a_branch2_1_conv = None
        return inception5a_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b15_2(self, maxpool4):
        inception5a_branch3_0_conv = getattr(self.inception5a.branch3, "0").conv(maxpool4)
        inception5a_branch3_0_relu = getattr(self.inception5a.branch3, "0").relu(inception5a_branch3_0_conv);  inception5a_branch3_0_conv = None
        inception5a_branch3_1_conv = getattr(self.inception5a.branch3, "1").conv(inception5a_branch3_0_relu);  inception5a_branch3_0_relu = None
        inception5a_branch3_1_relu = getattr(self.inception5a.branch3, "1").relu(inception5a_branch3_1_conv);  inception5a_branch3_1_conv = None
        return inception5a_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b15_3(self, maxpool4):
        inception5a_branch4_0 = getattr(self.inception5a.branch4, "0")(maxpool4);  maxpool4 = None
        inception5a_branch4_1_conv = getattr(self.inception5a.branch4, "1").conv(inception5a_branch4_0);  inception5a_branch4_0 = None
        inception5a_branch4_1_relu = getattr(self.inception5a.branch4, "1").relu(inception5a_branch4_1_conv);  inception5a_branch4_1_conv = None
        return inception5a_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b16(self, inception5a_branch1_relu, inception5a_branch2_1_relu, inception5a_branch3_1_relu, inception5a_branch4_1_relu):
        cat_7 = torch.cat([inception5a_branch1_relu, inception5a_branch2_1_relu, inception5a_branch3_1_relu, inception5a_branch4_1_relu], 1);  inception5a_branch1_relu = inception5a_branch2_1_relu = inception5a_branch3_1_relu = inception5a_branch4_1_relu = None
        return cat_7

    @ray.remote(scheduling_strategy="SPREAD")
    def b17_0(self, cat_7):
        inception5b_branch1_conv = self.inception5b.branch1.conv(cat_7)
        inception5b_branch1_relu = self.inception5b.branch1.relu(inception5b_branch1_conv);  inception5b_branch1_conv = None
        return inception5b_branch1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b17_1(self, cat_7):
        inception5b_branch2_0_conv = getattr(self.inception5b.branch2, "0").conv(cat_7)
        inception5b_branch2_0_relu = getattr(self.inception5b.branch2, "0").relu(inception5b_branch2_0_conv);  inception5b_branch2_0_conv = None
        inception5b_branch2_1_conv = getattr(self.inception5b.branch2, "1").conv(inception5b_branch2_0_relu);  inception5b_branch2_0_relu = None
        inception5b_branch2_1_relu = getattr(self.inception5b.branch2, "1").relu(inception5b_branch2_1_conv);  inception5b_branch2_1_conv = None
        return inception5b_branch2_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b17_2(self, cat_7):
        inception5b_branch3_0_conv = getattr(self.inception5b.branch3, "0").conv(cat_7)
        inception5b_branch3_0_relu = getattr(self.inception5b.branch3, "0").relu(inception5b_branch3_0_conv);  inception5b_branch3_0_conv = None
        inception5b_branch3_1_conv = getattr(self.inception5b.branch3, "1").conv(inception5b_branch3_0_relu);  inception5b_branch3_0_relu = None
        inception5b_branch3_1_relu = getattr(self.inception5b.branch3, "1").relu(inception5b_branch3_1_conv);  inception5b_branch3_1_conv = None
        return inception5b_branch3_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b17_3(self, cat_7):
        inception5b_branch4_0 = getattr(self.inception5b.branch4, "0")(cat_7);  cat_7 = None
        inception5b_branch4_1_conv = getattr(self.inception5b.branch4, "1").conv(inception5b_branch4_0);  inception5b_branch4_0 = None
        inception5b_branch4_1_relu = getattr(self.inception5b.branch4, "1").relu(inception5b_branch4_1_conv);  inception5b_branch4_1_conv = None
        return inception5b_branch4_1_relu

    @ray.remote(scheduling_strategy="SPREAD")
    def b18(self, inception5b_branch1_relu, inception5b_branch2_1_relu, inception5b_branch3_1_relu, inception5b_branch4_1_relu):
        cat_8 = torch.cat([inception5b_branch1_relu, inception5b_branch2_1_relu, inception5b_branch3_1_relu, inception5b_branch4_1_relu], 1);  inception5b_branch1_relu = inception5b_branch2_1_relu = inception5b_branch3_1_relu = inception5b_branch4_1_relu = None
        return cat_8

    @ray.remote(scheduling_strategy="SPREAD")
    def tail(self, cat_8):
        avgpool = self.avgpool(cat_8);  cat_8 = None
        flatten = torch.flatten(avgpool, 1);  avgpool = None
        dropout = self.dropout(flatten);  flatten = None
        fc = self.fc(dropout);  dropout = None
        return fc

    def forward(self, x):
        x = self.b0(x)
        b1_0 = self.b1_0.remote(self, x)
        b1_1 = self.b1_1.remote(self, x)
        b1_2 = self.b1_2.remote(self, x)
        b1_3 = self.b1_3.remote(self, x)
        x = self.b2.remote(self, b1_0, b1_1, b1_2, b1_3)
        b3_0 = self.b3_0.remote(self, x)
        b3_1 = self.b3_1.remote(self, x)
        b3_2 = self.b3_2.remote(self, x)
        b3_3 = self.b3_3.remote(self, x)
        x = self.b4.remote(self, b3_0, b3_1, b3_2, b3_3)
        b5_0 = self.b5_0.remote(self, x)
        b5_1 = self.b5_1.remote(self, x)
        b5_2 = self.b5_2.remote(self, x)
        b5_3 = self.b5_3.remote(self, x)
        x = self.b6.remote(self, b5_0, b5_1, b5_2, b5_3)
        b7_0 = self.b7_0.remote(self, x)
        b7_1 = self.b7_1.remote(self, x)
        b7_2 = self.b7_2.remote(self, x)
        b7_3 = self.b7_3.remote(self, x)
        x = self.b8.remote(self, b7_0, b7_1, b7_2, b7_3)
        b9_0 = self.b9_0.remote(self, x)
        b9_1 = self.b9_1.remote(self, x)
        b9_2 = self.b9_2.remote(self, x)
        b9_3 = self.b9_3.remote(self, x)
        x = self.b10.remote(self, b9_0, b9_1, b9_2, b9_3)
        b11_0 = self.b11_0.remote(self, x)
        b11_1 = self.b11_1.remote(self, x)
        b11_2 = self.b11_2.remote(self, x)
        b11_3 = self.b11_3.remote(self, x)
        x = self.b12.remote(self, b11_0, b11_1, b11_2, b11_3)
        b13_0 = self.b13_0.remote(self, x)
        b13_1 = self.b13_1.remote(self, x)
        b13_2 = self.b13_2.remote(self, x)
        b13_3 = self.b13_3.remote(self, x)
        x = self.b14.remote(self, b13_0, b13_1, b13_2, b13_3)
        b15_0 = self.b15_0.remote(self, x)
        b15_1 = self.b15_1.remote(self, x)
        b15_2 = self.b15_2.remote(self, x)
        b15_3 = self.b15_3.remote(self, x)
        x = self.b16.remote(self, b15_0, b15_1, b15_2, b15_3)
        b17_0 = self.b17_0.remote(self, x)
        b17_1 = self.b17_1.remote(self, x)
        b17_2 = self.b17_2.remote(self, x)
        b17_3 = self.b17_3.remote(self, x)
        x = self.b18.remote(self, b17_0, b17_1, b17_2, b17_3)
        x = self.tail.remote(self, x)
        return x


import time
import requests, warnings
import tarfile
import shutil, os
warnings.filterwarnings("ignore")
s = requests.Session()
s = requests.Session()
for i in range(10):
    try:
        res = s.get("https://10.101.15.6/storage/jcce/b9a64d47-da-47d0-9e37-0147f13f493c-googlenet.tar.gz", verify=False)
        print("下载成功")
        with open("googlenet.tar.gz", "wb") as f:
            f.write(res.content)
        data_folder = os.path.join("/tmp", "8888")
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder)
        os.mkdir(data_folder)
        tf = tarfile.open("googlenet.tar.gz")
        tf.extractall(data_folder)
        break
    except:
        print("下载失败, 正在重试...")
        continue
else:
    print("下载失败， 超过最大重试次数")
    exit()
all_data = []  # 数据路径列表
for p, _, files in os.walk(data_folder):
    if files:
        for f in files:
            all_data.append(os.path.join(p, f))
model = GoogLeNet()

t1 = time.time()
for p in all_data:
    x = torch.load(p)
    y = model.forward(x)
    print("current frame: %s %s"% (type(y), ray.get(y).shape))
t2 = time.time() - t1
print(f"inference result: {ray.get(y).shape}")
print(f"time: {t2}s")
#import time
#ray.init(address='auto')
#x = torch.randn(1, 1, 96, 96)
#model = GoogLeNet()
#t1 = time.time()
#for i in range(16):
#    y = model.forward(x)
#    print(f"current frame: {y}")
#print(f"inference result: {ray.get(y).shape}")
#t2 = time.time() - t1
#print(f"time： {t2}s")
