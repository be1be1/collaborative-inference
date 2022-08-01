import torch
from torch import nn
import time

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        output_size = []
        start1 = time.time()
        out = self.conv1(x)
        end1 = time.time()
        output_size.append(out.size())
        start2 = time.time()
        out = self.conv2(out)
        end2 = time.time()
        output_size.append(out.size())
        start3 = time.time()
        out = self.residual_block1(out)
        end3 = time.time()
        output_size.append(out.size())
        start4 = time.time()
        out = self.conv3(out)
        end4 = time.time()
        output_size.append(out.size())
        start5 = time.time()
        out = self.residual_block2(out)
        end5 = time.time()
        output_size.append(out.size())
        start6 = time.time()
        out = self.conv4(out)
        end6 = time.time()
        output_size.append(out.size())
        start7 = time.time()
        out = self.residual_block3(out)
        end7 = time.time()
        output_size.append(out.size())
        start8 = time.time()
        out = self.conv5(out)
        end8 = time.time()
        output_size.append(out.size())
        start9 = time.time()
        out = self.residual_block4(out)
        end9 = time.time()
        output_size.append(out.size())
        start10 = time.time()
        out = self.conv6(out)
        end10 = time.time()
        output_size.append(out.size())
        start11 = time.time()
        out = self.residual_block5(out)
        end11 = time.time()
        output_size.append(out.size())
        start12 = time.time()
        out = self.global_avg_pool(out)
        end12 = time.time()
        output_size.append(out.size())
        out = out.view(-1, 1024)
        start13 = time.time()
        out = self.fc(out)
        end13 = time.time()
        output_size.append(out.size())
        proc_time = [end1-start1, end2-start2, end3-start3, end4-start4, end5-start5, end6-start6, end7-start7, end8-start8, end9-start9, end10-start10, end11-start11, end12-start12, end13-start13]
        return out, proc_time, output_size

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)
