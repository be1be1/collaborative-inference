from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

__all__ = ['InceptionV4', 'inceptionv4']

pretrained_settings = {
    'inceptionv4': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),# Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        # adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=8)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def inceptionv4(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['inceptionv4'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionV4(num_classes=num_classes)
    return model


'''
TEST
Run this code with:
```
cd $HOME/pretrained-models.pytorch
python -m pretrainedmodels.inceptionv4
```
'''
if __name__ == '__main__':

    assert inceptionv4(num_classes=10, pretrained=None)
    print('success')
    assert inceptionv4(num_classes=1000, pretrained='imagenet')
    print('success')
    assert inceptionv4(num_classes=1001, pretrained='imagenet+background')
    print('success')

    # fail
    assert inceptionv4(num_classes=1000, pretrained='imagenet')


import ray
class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),# Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        # adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=8)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

#    def forward(self, input):
#        x = self.features(input)
#        x = self.logits(x)
#        return x

    def b0(self, input_1):
        features_0_conv = getattr(self.features, "0").conv(input_1);  input_1 = None
        features_0_bn = getattr(self.features, "0").bn(features_0_conv);  features_0_conv = None
        features_0_relu = getattr(self.features, "0").relu(features_0_bn);  features_0_bn = None
        features_1_conv = getattr(self.features, "1").conv(features_0_relu);  features_0_relu = None
        features_1_bn = getattr(self.features, "1").bn(features_1_conv);  features_1_conv = None
        features_1_relu = getattr(self.features, "1").relu(features_1_bn);  features_1_bn = None
        features_2_conv = getattr(self.features, "2").conv(features_1_relu);  features_1_relu = None
        features_2_bn = getattr(self.features, "2").bn(features_2_conv);  features_2_conv = None
        features_2_relu = getattr(self.features, "2").relu(features_2_bn);  features_2_bn = None
        return features_2_relu

    @ray.remote
    def b1_0(self, features_2_relu):
        features_3_maxpool = getattr(self.features, "3").maxpool(features_2_relu)
        return features_3_maxpool

    @ray.remote
    def b1_1(self, features_2_relu):
        features_3_conv_conv = getattr(self.features, "3").conv.conv(features_2_relu);  features_2_relu = None
        features_3_conv_bn = getattr(self.features, "3").conv.bn(features_3_conv_conv);  features_3_conv_conv = None
        features_3_conv_relu = getattr(self.features, "3").conv.relu(features_3_conv_bn);  features_3_conv_bn = None
        return features_3_conv_relu

    def b2(self, features_3_maxpool, features_3_conv_relu):
        cat = torch.cat((features_3_maxpool, features_3_conv_relu), 1);  features_3_maxpool = features_3_conv_relu = None
        return cat

    @ray.remote
    def b3_0(self, cat):
        features_4_branch0_0_conv = getattr(getattr(self.features, "4").branch0, "0").conv(cat)
        features_4_branch0_0_bn = getattr(getattr(self.features, "4").branch0, "0").bn(features_4_branch0_0_conv);  features_4_branch0_0_conv = None
        features_4_branch0_0_relu = getattr(getattr(self.features, "4").branch0, "0").relu(features_4_branch0_0_bn);  features_4_branch0_0_bn = None
        features_4_branch0_1_conv = getattr(getattr(self.features, "4").branch0, "1").conv(features_4_branch0_0_relu);  features_4_branch0_0_relu = None
        features_4_branch0_1_bn = getattr(getattr(self.features, "4").branch0, "1").bn(features_4_branch0_1_conv);  features_4_branch0_1_conv = None
        features_4_branch0_1_relu = getattr(getattr(self.features, "4").branch0, "1").relu(features_4_branch0_1_bn);  features_4_branch0_1_bn = None
        return features_4_branch0_1_relu

    @ray.remote
    def b3_1(self, cat):
        features_4_branch1_0_conv = getattr(getattr(self.features, "4").branch1, "0").conv(cat);  cat = None
        features_4_branch1_0_bn = getattr(getattr(self.features, "4").branch1, "0").bn(features_4_branch1_0_conv);  features_4_branch1_0_conv = None
        features_4_branch1_0_relu = getattr(getattr(self.features, "4").branch1, "0").relu(features_4_branch1_0_bn);  features_4_branch1_0_bn = None
        features_4_branch1_1_conv = getattr(getattr(self.features, "4").branch1, "1").conv(features_4_branch1_0_relu);  features_4_branch1_0_relu = None
        features_4_branch1_1_bn = getattr(getattr(self.features, "4").branch1, "1").bn(features_4_branch1_1_conv);  features_4_branch1_1_conv = None
        features_4_branch1_1_relu = getattr(getattr(self.features, "4").branch1, "1").relu(features_4_branch1_1_bn);  features_4_branch1_1_bn = None
        features_4_branch1_2_conv = getattr(getattr(self.features, "4").branch1, "2").conv(features_4_branch1_1_relu);  features_4_branch1_1_relu = None
        features_4_branch1_2_bn = getattr(getattr(self.features, "4").branch1, "2").bn(features_4_branch1_2_conv);  features_4_branch1_2_conv = None
        features_4_branch1_2_relu = getattr(getattr(self.features, "4").branch1, "2").relu(features_4_branch1_2_bn);  features_4_branch1_2_bn = None
        features_4_branch1_3_conv = getattr(getattr(self.features, "4").branch1, "3").conv(features_4_branch1_2_relu);  features_4_branch1_2_relu = None
        features_4_branch1_3_bn = getattr(getattr(self.features, "4").branch1, "3").bn(features_4_branch1_3_conv);  features_4_branch1_3_conv = None
        features_4_branch1_3_relu = getattr(getattr(self.features, "4").branch1, "3").relu(features_4_branch1_3_bn);  features_4_branch1_3_bn = None
        return features_4_branch1_3_relu

    def b4(self, features_4_branch0_1_relu, features_4_branch1_3_relu):
        cat_1 = torch.cat((features_4_branch0_1_relu, features_4_branch1_3_relu), 1);  features_4_branch0_1_relu = features_4_branch1_3_relu = None
        return cat_1

    @ray.remote
    def b5_0(self, cat_1):
        features_5_conv_conv = getattr(self.features, "5").conv.conv(cat_1)
        features_5_conv_bn = getattr(self.features, "5").conv.bn(features_5_conv_conv);  features_5_conv_conv = None
        features_5_conv_relu = getattr(self.features, "5").conv.relu(features_5_conv_bn);  features_5_conv_bn = None
        return features_5_conv_relu

    @ray.remote
    def b5_1(self, cat_1):
        features_5_maxpool = getattr(self.features, "5").maxpool(cat_1);  cat_1 = None
        return features_5_maxpool

    def b6(self, features_5_conv_relu, features_5_maxpool):
        cat_2 = torch.cat((features_5_conv_relu, features_5_maxpool), 1);  features_5_conv_relu = features_5_maxpool = None
        return cat_2

    @ray.remote
    def b7_0(self, cat_2):
        features_6_branch0_conv = getattr(self.features, "6").branch0.conv(cat_2)
        features_6_branch0_bn = getattr(self.features, "6").branch0.bn(features_6_branch0_conv);  features_6_branch0_conv = None
        features_6_branch0_relu = getattr(self.features, "6").branch0.relu(features_6_branch0_bn);  features_6_branch0_bn = None
        return features_6_branch0_relu

    @ray.remote
    def b7_1(self, cat_2):
        features_6_branch1_0_conv = getattr(getattr(self.features, "6").branch1, "0").conv(cat_2)
        features_6_branch1_0_bn = getattr(getattr(self.features, "6").branch1, "0").bn(features_6_branch1_0_conv);  features_6_branch1_0_conv = None
        features_6_branch1_0_relu = getattr(getattr(self.features, "6").branch1, "0").relu(features_6_branch1_0_bn);  features_6_branch1_0_bn = None
        features_6_branch1_1_conv = getattr(getattr(self.features, "6").branch1, "1").conv(features_6_branch1_0_relu);  features_6_branch1_0_relu = None
        features_6_branch1_1_bn = getattr(getattr(self.features, "6").branch1, "1").bn(features_6_branch1_1_conv);  features_6_branch1_1_conv = None
        features_6_branch1_1_relu = getattr(getattr(self.features, "6").branch1, "1").relu(features_6_branch1_1_bn);  features_6_branch1_1_bn = None
        return features_6_branch1_1_relu

    @ray.remote
    def b7_2(self, cat_2):
        features_6_branch2_0_conv = getattr(getattr(self.features, "6").branch2, "0").conv(cat_2)
        features_6_branch2_0_bn = getattr(getattr(self.features, "6").branch2, "0").bn(features_6_branch2_0_conv);  features_6_branch2_0_conv = None
        features_6_branch2_0_relu = getattr(getattr(self.features, "6").branch2, "0").relu(features_6_branch2_0_bn);  features_6_branch2_0_bn = None
        features_6_branch2_1_conv = getattr(getattr(self.features, "6").branch2, "1").conv(features_6_branch2_0_relu);  features_6_branch2_0_relu = None
        features_6_branch2_1_bn = getattr(getattr(self.features, "6").branch2, "1").bn(features_6_branch2_1_conv);  features_6_branch2_1_conv = None
        features_6_branch2_1_relu = getattr(getattr(self.features, "6").branch2, "1").relu(features_6_branch2_1_bn);  features_6_branch2_1_bn = None
        features_6_branch2_2_conv = getattr(getattr(self.features, "6").branch2, "2").conv(features_6_branch2_1_relu);  features_6_branch2_1_relu = None
        features_6_branch2_2_bn = getattr(getattr(self.features, "6").branch2, "2").bn(features_6_branch2_2_conv);  features_6_branch2_2_conv = None
        features_6_branch2_2_relu = getattr(getattr(self.features, "6").branch2, "2").relu(features_6_branch2_2_bn);  features_6_branch2_2_bn = None
        return features_6_branch2_2_relu

    @ray.remote
    def b7_3(self, cat_2):
        features_6_branch3_0 = getattr(getattr(self.features, "6").branch3, "0")(cat_2);  cat_2 = None
        features_6_branch3_1_conv = getattr(getattr(self.features, "6").branch3, "1").conv(features_6_branch3_0);  features_6_branch3_0 = None
        features_6_branch3_1_bn = getattr(getattr(self.features, "6").branch3, "1").bn(features_6_branch3_1_conv);  features_6_branch3_1_conv = None
        features_6_branch3_1_relu = getattr(getattr(self.features, "6").branch3, "1").relu(features_6_branch3_1_bn);  features_6_branch3_1_bn = None
        return features_6_branch3_1_relu

    def b8(self, features_6_branch0_relu, features_6_branch1_1_relu, features_6_branch2_2_relu, features_6_branch3_1_relu):
        cat_3 = torch.cat((features_6_branch0_relu, features_6_branch1_1_relu, features_6_branch2_2_relu, features_6_branch3_1_relu), 1);  features_6_branch0_relu = features_6_branch1_1_relu = features_6_branch2_2_relu = features_6_branch3_1_relu = None
        return cat_3

    @ray.remote
    def b9_0(self, cat_3):
        features_7_branch0_conv = getattr(self.features, "7").branch0.conv(cat_3)
        features_7_branch0_bn = getattr(self.features, "7").branch0.bn(features_7_branch0_conv);  features_7_branch0_conv = None
        features_7_branch0_relu = getattr(self.features, "7").branch0.relu(features_7_branch0_bn);  features_7_branch0_bn = None
        return features_7_branch0_relu

    @ray.remote
    def b9_1(self, cat_3):
        features_7_branch1_0_conv = getattr(getattr(self.features, "7").branch1, "0").conv(cat_3)
        features_7_branch1_0_bn = getattr(getattr(self.features, "7").branch1, "0").bn(features_7_branch1_0_conv);  features_7_branch1_0_conv = None
        features_7_branch1_0_relu = getattr(getattr(self.features, "7").branch1, "0").relu(features_7_branch1_0_bn);  features_7_branch1_0_bn = None
        features_7_branch1_1_conv = getattr(getattr(self.features, "7").branch1, "1").conv(features_7_branch1_0_relu);  features_7_branch1_0_relu = None
        features_7_branch1_1_bn = getattr(getattr(self.features, "7").branch1, "1").bn(features_7_branch1_1_conv);  features_7_branch1_1_conv = None
        features_7_branch1_1_relu = getattr(getattr(self.features, "7").branch1, "1").relu(features_7_branch1_1_bn);  features_7_branch1_1_bn = None
        return features_7_branch1_1_relu

    @ray.remote
    def b9_2(self, cat_3):
        features_7_branch2_0_conv = getattr(getattr(self.features, "7").branch2, "0").conv(cat_3)
        features_7_branch2_0_bn = getattr(getattr(self.features, "7").branch2, "0").bn(features_7_branch2_0_conv);  features_7_branch2_0_conv = None
        features_7_branch2_0_relu = getattr(getattr(self.features, "7").branch2, "0").relu(features_7_branch2_0_bn);  features_7_branch2_0_bn = None
        features_7_branch2_1_conv = getattr(getattr(self.features, "7").branch2, "1").conv(features_7_branch2_0_relu);  features_7_branch2_0_relu = None
        features_7_branch2_1_bn = getattr(getattr(self.features, "7").branch2, "1").bn(features_7_branch2_1_conv);  features_7_branch2_1_conv = None
        features_7_branch2_1_relu = getattr(getattr(self.features, "7").branch2, "1").relu(features_7_branch2_1_bn);  features_7_branch2_1_bn = None
        features_7_branch2_2_conv = getattr(getattr(self.features, "7").branch2, "2").conv(features_7_branch2_1_relu);  features_7_branch2_1_relu = None
        features_7_branch2_2_bn = getattr(getattr(self.features, "7").branch2, "2").bn(features_7_branch2_2_conv);  features_7_branch2_2_conv = None
        features_7_branch2_2_relu = getattr(getattr(self.features, "7").branch2, "2").relu(features_7_branch2_2_bn);  features_7_branch2_2_bn = None
        return features_7_branch2_2_relu

    @ray.remote
    def b9_3(self, cat_3):
        features_7_branch3_0 = getattr(getattr(self.features, "7").branch3, "0")(cat_3);  cat_3 = None
        features_7_branch3_1_conv = getattr(getattr(self.features, "7").branch3, "1").conv(features_7_branch3_0);  features_7_branch3_0 = None
        features_7_branch3_1_bn = getattr(getattr(self.features, "7").branch3, "1").bn(features_7_branch3_1_conv);  features_7_branch3_1_conv = None
        features_7_branch3_1_relu = getattr(getattr(self.features, "7").branch3, "1").relu(features_7_branch3_1_bn);  features_7_branch3_1_bn = None
        return features_7_branch3_1_relu

    def b10(self, features_7_branch0_relu, features_7_branch1_1_relu, features_7_branch2_2_relu, features_7_branch3_1_relu):
        cat_4 = torch.cat((features_7_branch0_relu, features_7_branch1_1_relu, features_7_branch2_2_relu, features_7_branch3_1_relu), 1);  features_7_branch0_relu = features_7_branch1_1_relu = features_7_branch2_2_relu = features_7_branch3_1_relu = None
        return cat_4

    @ray.remote
    def b11_0(self, cat_4):
        features_8_branch0_conv = getattr(self.features, "8").branch0.conv(cat_4)
        features_8_branch0_bn = getattr(self.features, "8").branch0.bn(features_8_branch0_conv);  features_8_branch0_conv = None
        features_8_branch0_relu = getattr(self.features, "8").branch0.relu(features_8_branch0_bn);  features_8_branch0_bn = None
        return features_8_branch0_relu

    @ray.remote
    def b11_1(self, cat_4):
        features_8_branch1_0_conv = getattr(getattr(self.features, "8").branch1, "0").conv(cat_4)
        features_8_branch1_0_bn = getattr(getattr(self.features, "8").branch1, "0").bn(features_8_branch1_0_conv);  features_8_branch1_0_conv = None
        features_8_branch1_0_relu = getattr(getattr(self.features, "8").branch1, "0").relu(features_8_branch1_0_bn);  features_8_branch1_0_bn = None
        features_8_branch1_1_conv = getattr(getattr(self.features, "8").branch1, "1").conv(features_8_branch1_0_relu);  features_8_branch1_0_relu = None
        features_8_branch1_1_bn = getattr(getattr(self.features, "8").branch1, "1").bn(features_8_branch1_1_conv);  features_8_branch1_1_conv = None
        features_8_branch1_1_relu = getattr(getattr(self.features, "8").branch1, "1").relu(features_8_branch1_1_bn);  features_8_branch1_1_bn = None
        return features_8_branch1_1_relu

    @ray.remote
    def b11_2(self, cat_4):
        features_8_branch2_0_conv = getattr(getattr(self.features, "8").branch2, "0").conv(cat_4)
        features_8_branch2_0_bn = getattr(getattr(self.features, "8").branch2, "0").bn(features_8_branch2_0_conv);  features_8_branch2_0_conv = None
        features_8_branch2_0_relu = getattr(getattr(self.features, "8").branch2, "0").relu(features_8_branch2_0_bn);  features_8_branch2_0_bn = None
        features_8_branch2_1_conv = getattr(getattr(self.features, "8").branch2, "1").conv(features_8_branch2_0_relu);  features_8_branch2_0_relu = None
        features_8_branch2_1_bn = getattr(getattr(self.features, "8").branch2, "1").bn(features_8_branch2_1_conv);  features_8_branch2_1_conv = None
        features_8_branch2_1_relu = getattr(getattr(self.features, "8").branch2, "1").relu(features_8_branch2_1_bn);  features_8_branch2_1_bn = None
        features_8_branch2_2_conv = getattr(getattr(self.features, "8").branch2, "2").conv(features_8_branch2_1_relu);  features_8_branch2_1_relu = None
        features_8_branch2_2_bn = getattr(getattr(self.features, "8").branch2, "2").bn(features_8_branch2_2_conv);  features_8_branch2_2_conv = None
        features_8_branch2_2_relu = getattr(getattr(self.features, "8").branch2, "2").relu(features_8_branch2_2_bn);  features_8_branch2_2_bn = None
        return features_8_branch2_2_relu

    @ray.remote
    def b11_3(self, cat_4):
        features_8_branch3_0 = getattr(getattr(self.features, "8").branch3, "0")(cat_4);  cat_4 = None
        features_8_branch3_1_conv = getattr(getattr(self.features, "8").branch3, "1").conv(features_8_branch3_0);  features_8_branch3_0 = None
        features_8_branch3_1_bn = getattr(getattr(self.features, "8").branch3, "1").bn(features_8_branch3_1_conv);  features_8_branch3_1_conv = None
        features_8_branch3_1_relu = getattr(getattr(self.features, "8").branch3, "1").relu(features_8_branch3_1_bn);  features_8_branch3_1_bn = None
        return features_8_branch3_1_relu

    def b12(self, features_8_branch0_relu, features_8_branch1_1_relu, features_8_branch2_2_relu, features_8_branch3_1_relu):
        cat_5 = torch.cat((features_8_branch0_relu, features_8_branch1_1_relu, features_8_branch2_2_relu, features_8_branch3_1_relu), 1);  features_8_branch0_relu = features_8_branch1_1_relu = features_8_branch2_2_relu = features_8_branch3_1_relu = None
        return cat_5

    @ray.remote
    def b13_0(self, cat_5):
        features_9_branch0_conv = getattr(self.features, "9").branch0.conv(cat_5)
        features_9_branch0_bn = getattr(self.features, "9").branch0.bn(features_9_branch0_conv);  features_9_branch0_conv = None
        features_9_branch0_relu = getattr(self.features, "9").branch0.relu(features_9_branch0_bn);  features_9_branch0_bn = None
        return features_9_branch0_relu

    @ray.remote
    def b13_1(self, cat_5):
        features_9_branch1_0_conv = getattr(getattr(self.features, "9").branch1, "0").conv(cat_5)
        features_9_branch1_0_bn = getattr(getattr(self.features, "9").branch1, "0").bn(features_9_branch1_0_conv);  features_9_branch1_0_conv = None
        features_9_branch1_0_relu = getattr(getattr(self.features, "9").branch1, "0").relu(features_9_branch1_0_bn);  features_9_branch1_0_bn = None
        features_9_branch1_1_conv = getattr(getattr(self.features, "9").branch1, "1").conv(features_9_branch1_0_relu);  features_9_branch1_0_relu = None
        features_9_branch1_1_bn = getattr(getattr(self.features, "9").branch1, "1").bn(features_9_branch1_1_conv);  features_9_branch1_1_conv = None
        features_9_branch1_1_relu = getattr(getattr(self.features, "9").branch1, "1").relu(features_9_branch1_1_bn);  features_9_branch1_1_bn = None
        return features_9_branch1_1_relu

    @ray.remote
    def b13_2(self, cat_5):
        features_9_branch2_0_conv = getattr(getattr(self.features, "9").branch2, "0").conv(cat_5)
        features_9_branch2_0_bn = getattr(getattr(self.features, "9").branch2, "0").bn(features_9_branch2_0_conv);  features_9_branch2_0_conv = None
        features_9_branch2_0_relu = getattr(getattr(self.features, "9").branch2, "0").relu(features_9_branch2_0_bn);  features_9_branch2_0_bn = None
        features_9_branch2_1_conv = getattr(getattr(self.features, "9").branch2, "1").conv(features_9_branch2_0_relu);  features_9_branch2_0_relu = None
        features_9_branch2_1_bn = getattr(getattr(self.features, "9").branch2, "1").bn(features_9_branch2_1_conv);  features_9_branch2_1_conv = None
        features_9_branch2_1_relu = getattr(getattr(self.features, "9").branch2, "1").relu(features_9_branch2_1_bn);  features_9_branch2_1_bn = None
        features_9_branch2_2_conv = getattr(getattr(self.features, "9").branch2, "2").conv(features_9_branch2_1_relu);  features_9_branch2_1_relu = None
        features_9_branch2_2_bn = getattr(getattr(self.features, "9").branch2, "2").bn(features_9_branch2_2_conv);  features_9_branch2_2_conv = None
        features_9_branch2_2_relu = getattr(getattr(self.features, "9").branch2, "2").relu(features_9_branch2_2_bn);  features_9_branch2_2_bn = None
        return features_9_branch2_2_relu

    @ray.remote
    def b13_3(self, cat_5):
        features_9_branch3_0 = getattr(getattr(self.features, "9").branch3, "0")(cat_5);  cat_5 = None
        features_9_branch3_1_conv = getattr(getattr(self.features, "9").branch3, "1").conv(features_9_branch3_0);  features_9_branch3_0 = None
        features_9_branch3_1_bn = getattr(getattr(self.features, "9").branch3, "1").bn(features_9_branch3_1_conv);  features_9_branch3_1_conv = None
        features_9_branch3_1_relu = getattr(getattr(self.features, "9").branch3, "1").relu(features_9_branch3_1_bn);  features_9_branch3_1_bn = None
        return features_9_branch3_1_relu

    def b14(self, features_9_branch0_relu, features_9_branch1_1_relu, features_9_branch2_2_relu, features_9_branch3_1_relu):
        cat_6 = torch.cat((features_9_branch0_relu, features_9_branch1_1_relu, features_9_branch2_2_relu, features_9_branch3_1_relu), 1);  features_9_branch0_relu = features_9_branch1_1_relu = features_9_branch2_2_relu = features_9_branch3_1_relu = None
        return cat_6

    @ray.remote
    def b15_0(self, cat_6):
        features_10_branch0_conv = getattr(self.features, "10").branch0.conv(cat_6)
        features_10_branch0_bn = getattr(self.features, "10").branch0.bn(features_10_branch0_conv);  features_10_branch0_conv = None
        features_10_branch0_relu = getattr(self.features, "10").branch0.relu(features_10_branch0_bn);  features_10_branch0_bn = None
        return features_10_branch0_relu

    @ray.remote
    def b15_1(self, cat_6):
        features_10_branch1_0_conv = getattr(getattr(self.features, "10").branch1, "0").conv(cat_6)
        features_10_branch1_0_bn = getattr(getattr(self.features, "10").branch1, "0").bn(features_10_branch1_0_conv);  features_10_branch1_0_conv = None
        features_10_branch1_0_relu = getattr(getattr(self.features, "10").branch1, "0").relu(features_10_branch1_0_bn);  features_10_branch1_0_bn = None
        features_10_branch1_1_conv = getattr(getattr(self.features, "10").branch1, "1").conv(features_10_branch1_0_relu);  features_10_branch1_0_relu = None
        features_10_branch1_1_bn = getattr(getattr(self.features, "10").branch1, "1").bn(features_10_branch1_1_conv);  features_10_branch1_1_conv = None
        features_10_branch1_1_relu = getattr(getattr(self.features, "10").branch1, "1").relu(features_10_branch1_1_bn);  features_10_branch1_1_bn = None
        features_10_branch1_2_conv = getattr(getattr(self.features, "10").branch1, "2").conv(features_10_branch1_1_relu);  features_10_branch1_1_relu = None
        features_10_branch1_2_bn = getattr(getattr(self.features, "10").branch1, "2").bn(features_10_branch1_2_conv);  features_10_branch1_2_conv = None
        features_10_branch1_2_relu = getattr(getattr(self.features, "10").branch1, "2").relu(features_10_branch1_2_bn);  features_10_branch1_2_bn = None
        return features_10_branch1_2_relu

    @ray.remote
    def b15_2(self, cat_6):
        features_10_branch2 = getattr(self.features, "10").branch2(cat_6);  cat_6 = None
        return features_10_branch2

    def b16(self, features_10_branch0_relu, features_10_branch1_2_relu, features_10_branch2):
        cat_7 = torch.cat((features_10_branch0_relu, features_10_branch1_2_relu, features_10_branch2), 1);  features_10_branch0_relu = features_10_branch1_2_relu = features_10_branch2 = None
        return cat_7

    @ray.remote
    def b17_0(self, cat_7):
        features_11_branch0_conv = getattr(self.features, "11").branch0.conv(cat_7)
        features_11_branch0_bn = getattr(self.features, "11").branch0.bn(features_11_branch0_conv);  features_11_branch0_conv = None
        features_11_branch0_relu = getattr(self.features, "11").branch0.relu(features_11_branch0_bn);  features_11_branch0_bn = None
        return features_11_branch0_relu

    @ray.remote
    def b17_1(self, cat_7):
        features_11_branch1_0_conv = getattr(getattr(self.features, "11").branch1, "0").conv(cat_7)
        features_11_branch1_0_bn = getattr(getattr(self.features, "11").branch1, "0").bn(features_11_branch1_0_conv);  features_11_branch1_0_conv = None
        features_11_branch1_0_relu = getattr(getattr(self.features, "11").branch1, "0").relu(features_11_branch1_0_bn);  features_11_branch1_0_bn = None
        features_11_branch1_1_conv = getattr(getattr(self.features, "11").branch1, "1").conv(features_11_branch1_0_relu);  features_11_branch1_0_relu = None
        features_11_branch1_1_bn = getattr(getattr(self.features, "11").branch1, "1").bn(features_11_branch1_1_conv);  features_11_branch1_1_conv = None
        features_11_branch1_1_relu = getattr(getattr(self.features, "11").branch1, "1").relu(features_11_branch1_1_bn);  features_11_branch1_1_bn = None
        features_11_branch1_2_conv = getattr(getattr(self.features, "11").branch1, "2").conv(features_11_branch1_1_relu);  features_11_branch1_1_relu = None
        features_11_branch1_2_bn = getattr(getattr(self.features, "11").branch1, "2").bn(features_11_branch1_2_conv);  features_11_branch1_2_conv = None
        features_11_branch1_2_relu = getattr(getattr(self.features, "11").branch1, "2").relu(features_11_branch1_2_bn);  features_11_branch1_2_bn = None
        return features_11_branch1_2_relu

    @ray.remote
    def b17_2(self, cat_7):
        features_11_branch2_0_conv = getattr(getattr(self.features, "11").branch2, "0").conv(cat_7)
        features_11_branch2_0_bn = getattr(getattr(self.features, "11").branch2, "0").bn(features_11_branch2_0_conv);  features_11_branch2_0_conv = None
        features_11_branch2_0_relu = getattr(getattr(self.features, "11").branch2, "0").relu(features_11_branch2_0_bn);  features_11_branch2_0_bn = None
        features_11_branch2_1_conv = getattr(getattr(self.features, "11").branch2, "1").conv(features_11_branch2_0_relu);  features_11_branch2_0_relu = None
        features_11_branch2_1_bn = getattr(getattr(self.features, "11").branch2, "1").bn(features_11_branch2_1_conv);  features_11_branch2_1_conv = None
        features_11_branch2_1_relu = getattr(getattr(self.features, "11").branch2, "1").relu(features_11_branch2_1_bn);  features_11_branch2_1_bn = None
        features_11_branch2_2_conv = getattr(getattr(self.features, "11").branch2, "2").conv(features_11_branch2_1_relu);  features_11_branch2_1_relu = None
        features_11_branch2_2_bn = getattr(getattr(self.features, "11").branch2, "2").bn(features_11_branch2_2_conv);  features_11_branch2_2_conv = None
        features_11_branch2_2_relu = getattr(getattr(self.features, "11").branch2, "2").relu(features_11_branch2_2_bn);  features_11_branch2_2_bn = None
        features_11_branch2_3_conv = getattr(getattr(self.features, "11").branch2, "3").conv(features_11_branch2_2_relu);  features_11_branch2_2_relu = None
        features_11_branch2_3_bn = getattr(getattr(self.features, "11").branch2, "3").bn(features_11_branch2_3_conv);  features_11_branch2_3_conv = None
        features_11_branch2_3_relu = getattr(getattr(self.features, "11").branch2, "3").relu(features_11_branch2_3_bn);  features_11_branch2_3_bn = None
        features_11_branch2_4_conv = getattr(getattr(self.features, "11").branch2, "4").conv(features_11_branch2_3_relu);  features_11_branch2_3_relu = None
        features_11_branch2_4_bn = getattr(getattr(self.features, "11").branch2, "4").bn(features_11_branch2_4_conv);  features_11_branch2_4_conv = None
        features_11_branch2_4_relu = getattr(getattr(self.features, "11").branch2, "4").relu(features_11_branch2_4_bn);  features_11_branch2_4_bn = None
        return features_11_branch2_4_relu

    @ray.remote
    def b17_3(self, cat_7):
        features_11_branch3_0 = getattr(getattr(self.features, "11").branch3, "0")(cat_7);  cat_7 = None
        features_11_branch3_1_conv = getattr(getattr(self.features, "11").branch3, "1").conv(features_11_branch3_0);  features_11_branch3_0 = None
        features_11_branch3_1_bn = getattr(getattr(self.features, "11").branch3, "1").bn(features_11_branch3_1_conv);  features_11_branch3_1_conv = None
        features_11_branch3_1_relu = getattr(getattr(self.features, "11").branch3, "1").relu(features_11_branch3_1_bn);  features_11_branch3_1_bn = None
        return features_11_branch3_1_relu

    def b18(self, features_11_branch0_relu, features_11_branch1_2_relu, features_11_branch2_4_relu, features_11_branch3_1_relu):
        cat_8 = torch.cat((features_11_branch0_relu, features_11_branch1_2_relu, features_11_branch2_4_relu, features_11_branch3_1_relu), 1);  features_11_branch0_relu = features_11_branch1_2_relu = features_11_branch2_4_relu = features_11_branch3_1_relu = None
        return cat_8

    @ray.remote
    def b19_0(self, cat_8):
        features_12_branch0_conv = getattr(self.features, "12").branch0.conv(cat_8)
        features_12_branch0_bn = getattr(self.features, "12").branch0.bn(features_12_branch0_conv);  features_12_branch0_conv = None
        features_12_branch0_relu = getattr(self.features, "12").branch0.relu(features_12_branch0_bn);  features_12_branch0_bn = None
        return features_12_branch0_relu

    @ray.remote
    def b19_1(self, cat_8):
        features_12_branch1_0_conv = getattr(getattr(self.features, "12").branch1, "0").conv(cat_8)
        features_12_branch1_0_bn = getattr(getattr(self.features, "12").branch1, "0").bn(features_12_branch1_0_conv);  features_12_branch1_0_conv = None
        features_12_branch1_0_relu = getattr(getattr(self.features, "12").branch1, "0").relu(features_12_branch1_0_bn);  features_12_branch1_0_bn = None
        features_12_branch1_1_conv = getattr(getattr(self.features, "12").branch1, "1").conv(features_12_branch1_0_relu);  features_12_branch1_0_relu = None
        features_12_branch1_1_bn = getattr(getattr(self.features, "12").branch1, "1").bn(features_12_branch1_1_conv);  features_12_branch1_1_conv = None
        features_12_branch1_1_relu = getattr(getattr(self.features, "12").branch1, "1").relu(features_12_branch1_1_bn);  features_12_branch1_1_bn = None
        features_12_branch1_2_conv = getattr(getattr(self.features, "12").branch1, "2").conv(features_12_branch1_1_relu);  features_12_branch1_1_relu = None
        features_12_branch1_2_bn = getattr(getattr(self.features, "12").branch1, "2").bn(features_12_branch1_2_conv);  features_12_branch1_2_conv = None
        features_12_branch1_2_relu = getattr(getattr(self.features, "12").branch1, "2").relu(features_12_branch1_2_bn);  features_12_branch1_2_bn = None
        return features_12_branch1_2_relu

    @ray.remote
    def b19_2(self, cat_8):
        features_12_branch2_0_conv = getattr(getattr(self.features, "12").branch2, "0").conv(cat_8)
        features_12_branch2_0_bn = getattr(getattr(self.features, "12").branch2, "0").bn(features_12_branch2_0_conv);  features_12_branch2_0_conv = None
        features_12_branch2_0_relu = getattr(getattr(self.features, "12").branch2, "0").relu(features_12_branch2_0_bn);  features_12_branch2_0_bn = None
        features_12_branch2_1_conv = getattr(getattr(self.features, "12").branch2, "1").conv(features_12_branch2_0_relu);  features_12_branch2_0_relu = None
        features_12_branch2_1_bn = getattr(getattr(self.features, "12").branch2, "1").bn(features_12_branch2_1_conv);  features_12_branch2_1_conv = None
        features_12_branch2_1_relu = getattr(getattr(self.features, "12").branch2, "1").relu(features_12_branch2_1_bn);  features_12_branch2_1_bn = None
        features_12_branch2_2_conv = getattr(getattr(self.features, "12").branch2, "2").conv(features_12_branch2_1_relu);  features_12_branch2_1_relu = None
        features_12_branch2_2_bn = getattr(getattr(self.features, "12").branch2, "2").bn(features_12_branch2_2_conv);  features_12_branch2_2_conv = None
        features_12_branch2_2_relu = getattr(getattr(self.features, "12").branch2, "2").relu(features_12_branch2_2_bn);  features_12_branch2_2_bn = None
        features_12_branch2_3_conv = getattr(getattr(self.features, "12").branch2, "3").conv(features_12_branch2_2_relu);  features_12_branch2_2_relu = None
        features_12_branch2_3_bn = getattr(getattr(self.features, "12").branch2, "3").bn(features_12_branch2_3_conv);  features_12_branch2_3_conv = None
        features_12_branch2_3_relu = getattr(getattr(self.features, "12").branch2, "3").relu(features_12_branch2_3_bn);  features_12_branch2_3_bn = None
        features_12_branch2_4_conv = getattr(getattr(self.features, "12").branch2, "4").conv(features_12_branch2_3_relu);  features_12_branch2_3_relu = None
        features_12_branch2_4_bn = getattr(getattr(self.features, "12").branch2, "4").bn(features_12_branch2_4_conv);  features_12_branch2_4_conv = None
        features_12_branch2_4_relu = getattr(getattr(self.features, "12").branch2, "4").relu(features_12_branch2_4_bn);  features_12_branch2_4_bn = None
        return features_12_branch2_4_relu

    @ray.remote
    def b19_3(self, cat_8):
        features_12_branch3_0 = getattr(getattr(self.features, "12").branch3, "0")(cat_8);  cat_8 = None
        features_12_branch3_1_conv = getattr(getattr(self.features, "12").branch3, "1").conv(features_12_branch3_0);  features_12_branch3_0 = None
        features_12_branch3_1_bn = getattr(getattr(self.features, "12").branch3, "1").bn(features_12_branch3_1_conv);  features_12_branch3_1_conv = None
        features_12_branch3_1_relu = getattr(getattr(self.features, "12").branch3, "1").relu(features_12_branch3_1_bn);  features_12_branch3_1_bn = None
        return features_12_branch3_1_relu

    def b20(self, features_12_branch0_relu, features_12_branch1_2_relu, features_12_branch2_4_relu, features_12_branch3_1_relu):
        cat_9 = torch.cat((features_12_branch0_relu, features_12_branch1_2_relu, features_12_branch2_4_relu, features_12_branch3_1_relu), 1);  features_12_branch0_relu = features_12_branch1_2_relu = features_12_branch2_4_relu = features_12_branch3_1_relu = None
        return cat_9

    @ray.remote
    def b21_0(self, cat_9):
        features_13_branch0_conv = getattr(self.features, "13").branch0.conv(cat_9)
        features_13_branch0_bn = getattr(self.features, "13").branch0.bn(features_13_branch0_conv);  features_13_branch0_conv = None
        features_13_branch0_relu = getattr(self.features, "13").branch0.relu(features_13_branch0_bn);  features_13_branch0_bn = None
        return features_13_branch0_relu

    @ray.remote
    def b21_1(self, cat_9):
        features_13_branch1_0_conv = getattr(getattr(self.features, "13").branch1, "0").conv(cat_9)
        features_13_branch1_0_bn = getattr(getattr(self.features, "13").branch1, "0").bn(features_13_branch1_0_conv);  features_13_branch1_0_conv = None
        features_13_branch1_0_relu = getattr(getattr(self.features, "13").branch1, "0").relu(features_13_branch1_0_bn);  features_13_branch1_0_bn = None
        features_13_branch1_1_conv = getattr(getattr(self.features, "13").branch1, "1").conv(features_13_branch1_0_relu);  features_13_branch1_0_relu = None
        features_13_branch1_1_bn = getattr(getattr(self.features, "13").branch1, "1").bn(features_13_branch1_1_conv);  features_13_branch1_1_conv = None
        features_13_branch1_1_relu = getattr(getattr(self.features, "13").branch1, "1").relu(features_13_branch1_1_bn);  features_13_branch1_1_bn = None
        features_13_branch1_2_conv = getattr(getattr(self.features, "13").branch1, "2").conv(features_13_branch1_1_relu);  features_13_branch1_1_relu = None
        features_13_branch1_2_bn = getattr(getattr(self.features, "13").branch1, "2").bn(features_13_branch1_2_conv);  features_13_branch1_2_conv = None
        features_13_branch1_2_relu = getattr(getattr(self.features, "13").branch1, "2").relu(features_13_branch1_2_bn);  features_13_branch1_2_bn = None
        return features_13_branch1_2_relu

    @ray.remote
    def b21_2(self, cat_9):
        features_13_branch2_0_conv = getattr(getattr(self.features, "13").branch2, "0").conv(cat_9)
        features_13_branch2_0_bn = getattr(getattr(self.features, "13").branch2, "0").bn(features_13_branch2_0_conv);  features_13_branch2_0_conv = None
        features_13_branch2_0_relu = getattr(getattr(self.features, "13").branch2, "0").relu(features_13_branch2_0_bn);  features_13_branch2_0_bn = None
        features_13_branch2_1_conv = getattr(getattr(self.features, "13").branch2, "1").conv(features_13_branch2_0_relu);  features_13_branch2_0_relu = None
        features_13_branch2_1_bn = getattr(getattr(self.features, "13").branch2, "1").bn(features_13_branch2_1_conv);  features_13_branch2_1_conv = None
        features_13_branch2_1_relu = getattr(getattr(self.features, "13").branch2, "1").relu(features_13_branch2_1_bn);  features_13_branch2_1_bn = None
        features_13_branch2_2_conv = getattr(getattr(self.features, "13").branch2, "2").conv(features_13_branch2_1_relu);  features_13_branch2_1_relu = None
        features_13_branch2_2_bn = getattr(getattr(self.features, "13").branch2, "2").bn(features_13_branch2_2_conv);  features_13_branch2_2_conv = None
        features_13_branch2_2_relu = getattr(getattr(self.features, "13").branch2, "2").relu(features_13_branch2_2_bn);  features_13_branch2_2_bn = None
        features_13_branch2_3_conv = getattr(getattr(self.features, "13").branch2, "3").conv(features_13_branch2_2_relu);  features_13_branch2_2_relu = None
        features_13_branch2_3_bn = getattr(getattr(self.features, "13").branch2, "3").bn(features_13_branch2_3_conv);  features_13_branch2_3_conv = None
        features_13_branch2_3_relu = getattr(getattr(self.features, "13").branch2, "3").relu(features_13_branch2_3_bn);  features_13_branch2_3_bn = None
        features_13_branch2_4_conv = getattr(getattr(self.features, "13").branch2, "4").conv(features_13_branch2_3_relu);  features_13_branch2_3_relu = None
        features_13_branch2_4_bn = getattr(getattr(self.features, "13").branch2, "4").bn(features_13_branch2_4_conv);  features_13_branch2_4_conv = None
        features_13_branch2_4_relu = getattr(getattr(self.features, "13").branch2, "4").relu(features_13_branch2_4_bn);  features_13_branch2_4_bn = None
        return features_13_branch2_4_relu

    @ray.remote
    def b21_3(self, cat_9):
        features_13_branch3_0 = getattr(getattr(self.features, "13").branch3, "0")(cat_9);  cat_9 = None
        features_13_branch3_1_conv = getattr(getattr(self.features, "13").branch3, "1").conv(features_13_branch3_0);  features_13_branch3_0 = None
        features_13_branch3_1_bn = getattr(getattr(self.features, "13").branch3, "1").bn(features_13_branch3_1_conv);  features_13_branch3_1_conv = None
        features_13_branch3_1_relu = getattr(getattr(self.features, "13").branch3, "1").relu(features_13_branch3_1_bn);  features_13_branch3_1_bn = None
        return features_13_branch3_1_relu

    def b22(self, features_13_branch0_relu, features_13_branch1_2_relu, features_13_branch2_4_relu, features_13_branch3_1_relu):
        cat_10 = torch.cat((features_13_branch0_relu, features_13_branch1_2_relu, features_13_branch2_4_relu, features_13_branch3_1_relu), 1);  features_13_branch0_relu = features_13_branch1_2_relu = features_13_branch2_4_relu = features_13_branch3_1_relu = None
        return cat_10

    @ray.remote
    def b23_0(self, cat_10):
        features_14_branch0_conv = getattr(self.features, "14").branch0.conv(cat_10)
        features_14_branch0_bn = getattr(self.features, "14").branch0.bn(features_14_branch0_conv);  features_14_branch0_conv = None
        features_14_branch0_relu = getattr(self.features, "14").branch0.relu(features_14_branch0_bn);  features_14_branch0_bn = None
        return features_14_branch0_relu

    @ray.remote
    def b23_1(self, cat_10):
        features_14_branch1_0_conv = getattr(getattr(self.features, "14").branch1, "0").conv(cat_10)
        features_14_branch1_0_bn = getattr(getattr(self.features, "14").branch1, "0").bn(features_14_branch1_0_conv);  features_14_branch1_0_conv = None
        features_14_branch1_0_relu = getattr(getattr(self.features, "14").branch1, "0").relu(features_14_branch1_0_bn);  features_14_branch1_0_bn = None
        features_14_branch1_1_conv = getattr(getattr(self.features, "14").branch1, "1").conv(features_14_branch1_0_relu);  features_14_branch1_0_relu = None
        features_14_branch1_1_bn = getattr(getattr(self.features, "14").branch1, "1").bn(features_14_branch1_1_conv);  features_14_branch1_1_conv = None
        features_14_branch1_1_relu = getattr(getattr(self.features, "14").branch1, "1").relu(features_14_branch1_1_bn);  features_14_branch1_1_bn = None
        features_14_branch1_2_conv = getattr(getattr(self.features, "14").branch1, "2").conv(features_14_branch1_1_relu);  features_14_branch1_1_relu = None
        features_14_branch1_2_bn = getattr(getattr(self.features, "14").branch1, "2").bn(features_14_branch1_2_conv);  features_14_branch1_2_conv = None
        features_14_branch1_2_relu = getattr(getattr(self.features, "14").branch1, "2").relu(features_14_branch1_2_bn);  features_14_branch1_2_bn = None
        return features_14_branch1_2_relu

    @ray.remote
    def b23_2(self, cat_10):
        features_14_branch2_0_conv = getattr(getattr(self.features, "14").branch2, "0").conv(cat_10)
        features_14_branch2_0_bn = getattr(getattr(self.features, "14").branch2, "0").bn(features_14_branch2_0_conv);  features_14_branch2_0_conv = None
        features_14_branch2_0_relu = getattr(getattr(self.features, "14").branch2, "0").relu(features_14_branch2_0_bn);  features_14_branch2_0_bn = None
        features_14_branch2_1_conv = getattr(getattr(self.features, "14").branch2, "1").conv(features_14_branch2_0_relu);  features_14_branch2_0_relu = None
        features_14_branch2_1_bn = getattr(getattr(self.features, "14").branch2, "1").bn(features_14_branch2_1_conv);  features_14_branch2_1_conv = None
        features_14_branch2_1_relu = getattr(getattr(self.features, "14").branch2, "1").relu(features_14_branch2_1_bn);  features_14_branch2_1_bn = None
        features_14_branch2_2_conv = getattr(getattr(self.features, "14").branch2, "2").conv(features_14_branch2_1_relu);  features_14_branch2_1_relu = None
        features_14_branch2_2_bn = getattr(getattr(self.features, "14").branch2, "2").bn(features_14_branch2_2_conv);  features_14_branch2_2_conv = None
        features_14_branch2_2_relu = getattr(getattr(self.features, "14").branch2, "2").relu(features_14_branch2_2_bn);  features_14_branch2_2_bn = None
        features_14_branch2_3_conv = getattr(getattr(self.features, "14").branch2, "3").conv(features_14_branch2_2_relu);  features_14_branch2_2_relu = None
        features_14_branch2_3_bn = getattr(getattr(self.features, "14").branch2, "3").bn(features_14_branch2_3_conv);  features_14_branch2_3_conv = None
        features_14_branch2_3_relu = getattr(getattr(self.features, "14").branch2, "3").relu(features_14_branch2_3_bn);  features_14_branch2_3_bn = None
        features_14_branch2_4_conv = getattr(getattr(self.features, "14").branch2, "4").conv(features_14_branch2_3_relu);  features_14_branch2_3_relu = None
        features_14_branch2_4_bn = getattr(getattr(self.features, "14").branch2, "4").bn(features_14_branch2_4_conv);  features_14_branch2_4_conv = None
        features_14_branch2_4_relu = getattr(getattr(self.features, "14").branch2, "4").relu(features_14_branch2_4_bn);  features_14_branch2_4_bn = None
        return features_14_branch2_4_relu

    @ray.remote
    def b23_3(self, cat_10):
        features_14_branch3_0 = getattr(getattr(self.features, "14").branch3, "0")(cat_10);  cat_10 = None
        features_14_branch3_1_conv = getattr(getattr(self.features, "14").branch3, "1").conv(features_14_branch3_0);  features_14_branch3_0 = None
        features_14_branch3_1_bn = getattr(getattr(self.features, "14").branch3, "1").bn(features_14_branch3_1_conv);  features_14_branch3_1_conv = None
        features_14_branch3_1_relu = getattr(getattr(self.features, "14").branch3, "1").relu(features_14_branch3_1_bn);  features_14_branch3_1_bn = None
        return features_14_branch3_1_relu

    def b24(self, features_14_branch0_relu, features_14_branch1_2_relu, features_14_branch2_4_relu, features_14_branch3_1_relu):
        cat_11 = torch.cat((features_14_branch0_relu, features_14_branch1_2_relu, features_14_branch2_4_relu, features_14_branch3_1_relu), 1);  features_14_branch0_relu = features_14_branch1_2_relu = features_14_branch2_4_relu = features_14_branch3_1_relu = None
        return cat_11

    @ray.remote
    def b25_0(self, cat_11):
        features_15_branch0_conv = getattr(self.features, "15").branch0.conv(cat_11)
        features_15_branch0_bn = getattr(self.features, "15").branch0.bn(features_15_branch0_conv);  features_15_branch0_conv = None
        features_15_branch0_relu = getattr(self.features, "15").branch0.relu(features_15_branch0_bn);  features_15_branch0_bn = None
        return features_15_branch0_relu

    @ray.remote
    def b25_1(self, cat_11):
        features_15_branch1_0_conv = getattr(getattr(self.features, "15").branch1, "0").conv(cat_11)
        features_15_branch1_0_bn = getattr(getattr(self.features, "15").branch1, "0").bn(features_15_branch1_0_conv);  features_15_branch1_0_conv = None
        features_15_branch1_0_relu = getattr(getattr(self.features, "15").branch1, "0").relu(features_15_branch1_0_bn);  features_15_branch1_0_bn = None
        features_15_branch1_1_conv = getattr(getattr(self.features, "15").branch1, "1").conv(features_15_branch1_0_relu);  features_15_branch1_0_relu = None
        features_15_branch1_1_bn = getattr(getattr(self.features, "15").branch1, "1").bn(features_15_branch1_1_conv);  features_15_branch1_1_conv = None
        features_15_branch1_1_relu = getattr(getattr(self.features, "15").branch1, "1").relu(features_15_branch1_1_bn);  features_15_branch1_1_bn = None
        features_15_branch1_2_conv = getattr(getattr(self.features, "15").branch1, "2").conv(features_15_branch1_1_relu);  features_15_branch1_1_relu = None
        features_15_branch1_2_bn = getattr(getattr(self.features, "15").branch1, "2").bn(features_15_branch1_2_conv);  features_15_branch1_2_conv = None
        features_15_branch1_2_relu = getattr(getattr(self.features, "15").branch1, "2").relu(features_15_branch1_2_bn);  features_15_branch1_2_bn = None
        return features_15_branch1_2_relu

    @ray.remote
    def b25_2(self, cat_11):
        features_15_branch2_0_conv = getattr(getattr(self.features, "15").branch2, "0").conv(cat_11)
        features_15_branch2_0_bn = getattr(getattr(self.features, "15").branch2, "0").bn(features_15_branch2_0_conv);  features_15_branch2_0_conv = None
        features_15_branch2_0_relu = getattr(getattr(self.features, "15").branch2, "0").relu(features_15_branch2_0_bn);  features_15_branch2_0_bn = None
        features_15_branch2_1_conv = getattr(getattr(self.features, "15").branch2, "1").conv(features_15_branch2_0_relu);  features_15_branch2_0_relu = None
        features_15_branch2_1_bn = getattr(getattr(self.features, "15").branch2, "1").bn(features_15_branch2_1_conv);  features_15_branch2_1_conv = None
        features_15_branch2_1_relu = getattr(getattr(self.features, "15").branch2, "1").relu(features_15_branch2_1_bn);  features_15_branch2_1_bn = None
        features_15_branch2_2_conv = getattr(getattr(self.features, "15").branch2, "2").conv(features_15_branch2_1_relu);  features_15_branch2_1_relu = None
        features_15_branch2_2_bn = getattr(getattr(self.features, "15").branch2, "2").bn(features_15_branch2_2_conv);  features_15_branch2_2_conv = None
        features_15_branch2_2_relu = getattr(getattr(self.features, "15").branch2, "2").relu(features_15_branch2_2_bn);  features_15_branch2_2_bn = None
        features_15_branch2_3_conv = getattr(getattr(self.features, "15").branch2, "3").conv(features_15_branch2_2_relu);  features_15_branch2_2_relu = None
        features_15_branch2_3_bn = getattr(getattr(self.features, "15").branch2, "3").bn(features_15_branch2_3_conv);  features_15_branch2_3_conv = None
        features_15_branch2_3_relu = getattr(getattr(self.features, "15").branch2, "3").relu(features_15_branch2_3_bn);  features_15_branch2_3_bn = None
        features_15_branch2_4_conv = getattr(getattr(self.features, "15").branch2, "4").conv(features_15_branch2_3_relu);  features_15_branch2_3_relu = None
        features_15_branch2_4_bn = getattr(getattr(self.features, "15").branch2, "4").bn(features_15_branch2_4_conv);  features_15_branch2_4_conv = None
        features_15_branch2_4_relu = getattr(getattr(self.features, "15").branch2, "4").relu(features_15_branch2_4_bn);  features_15_branch2_4_bn = None
        return features_15_branch2_4_relu

    @ray.remote
    def b25_3(self, cat_11):
        features_15_branch3_0 = getattr(getattr(self.features, "15").branch3, "0")(cat_11);  cat_11 = None
        features_15_branch3_1_conv = getattr(getattr(self.features, "15").branch3, "1").conv(features_15_branch3_0);  features_15_branch3_0 = None
        features_15_branch3_1_bn = getattr(getattr(self.features, "15").branch3, "1").bn(features_15_branch3_1_conv);  features_15_branch3_1_conv = None
        features_15_branch3_1_relu = getattr(getattr(self.features, "15").branch3, "1").relu(features_15_branch3_1_bn);  features_15_branch3_1_bn = None
        return features_15_branch3_1_relu

    def b26(self, features_15_branch0_relu, features_15_branch1_2_relu, features_15_branch2_4_relu, features_15_branch3_1_relu):
        cat_12 = torch.cat((features_15_branch0_relu, features_15_branch1_2_relu, features_15_branch2_4_relu, features_15_branch3_1_relu), 1);  features_15_branch0_relu = features_15_branch1_2_relu = features_15_branch2_4_relu = features_15_branch3_1_relu = None
        return cat_12

    @ray.remote
    def b27_0(self, cat_12):
        features_16_branch0_conv = getattr(self.features, "16").branch0.conv(cat_12)
        features_16_branch0_bn = getattr(self.features, "16").branch0.bn(features_16_branch0_conv);  features_16_branch0_conv = None
        features_16_branch0_relu = getattr(self.features, "16").branch0.relu(features_16_branch0_bn);  features_16_branch0_bn = None
        return features_16_branch0_relu

    @ray.remote
    def b27_1(self, cat_12):
        features_16_branch1_0_conv = getattr(getattr(self.features, "16").branch1, "0").conv(cat_12)
        features_16_branch1_0_bn = getattr(getattr(self.features, "16").branch1, "0").bn(features_16_branch1_0_conv);  features_16_branch1_0_conv = None
        features_16_branch1_0_relu = getattr(getattr(self.features, "16").branch1, "0").relu(features_16_branch1_0_bn);  features_16_branch1_0_bn = None
        features_16_branch1_1_conv = getattr(getattr(self.features, "16").branch1, "1").conv(features_16_branch1_0_relu);  features_16_branch1_0_relu = None
        features_16_branch1_1_bn = getattr(getattr(self.features, "16").branch1, "1").bn(features_16_branch1_1_conv);  features_16_branch1_1_conv = None
        features_16_branch1_1_relu = getattr(getattr(self.features, "16").branch1, "1").relu(features_16_branch1_1_bn);  features_16_branch1_1_bn = None
        features_16_branch1_2_conv = getattr(getattr(self.features, "16").branch1, "2").conv(features_16_branch1_1_relu);  features_16_branch1_1_relu = None
        features_16_branch1_2_bn = getattr(getattr(self.features, "16").branch1, "2").bn(features_16_branch1_2_conv);  features_16_branch1_2_conv = None
        features_16_branch1_2_relu = getattr(getattr(self.features, "16").branch1, "2").relu(features_16_branch1_2_bn);  features_16_branch1_2_bn = None
        return features_16_branch1_2_relu

    @ray.remote
    def b27_2(self, cat_12):
        features_16_branch2_0_conv = getattr(getattr(self.features, "16").branch2, "0").conv(cat_12)
        features_16_branch2_0_bn = getattr(getattr(self.features, "16").branch2, "0").bn(features_16_branch2_0_conv);  features_16_branch2_0_conv = None
        features_16_branch2_0_relu = getattr(getattr(self.features, "16").branch2, "0").relu(features_16_branch2_0_bn);  features_16_branch2_0_bn = None
        features_16_branch2_1_conv = getattr(getattr(self.features, "16").branch2, "1").conv(features_16_branch2_0_relu);  features_16_branch2_0_relu = None
        features_16_branch2_1_bn = getattr(getattr(self.features, "16").branch2, "1").bn(features_16_branch2_1_conv);  features_16_branch2_1_conv = None
        features_16_branch2_1_relu = getattr(getattr(self.features, "16").branch2, "1").relu(features_16_branch2_1_bn);  features_16_branch2_1_bn = None
        features_16_branch2_2_conv = getattr(getattr(self.features, "16").branch2, "2").conv(features_16_branch2_1_relu);  features_16_branch2_1_relu = None
        features_16_branch2_2_bn = getattr(getattr(self.features, "16").branch2, "2").bn(features_16_branch2_2_conv);  features_16_branch2_2_conv = None
        features_16_branch2_2_relu = getattr(getattr(self.features, "16").branch2, "2").relu(features_16_branch2_2_bn);  features_16_branch2_2_bn = None
        features_16_branch2_3_conv = getattr(getattr(self.features, "16").branch2, "3").conv(features_16_branch2_2_relu);  features_16_branch2_2_relu = None
        features_16_branch2_3_bn = getattr(getattr(self.features, "16").branch2, "3").bn(features_16_branch2_3_conv);  features_16_branch2_3_conv = None
        features_16_branch2_3_relu = getattr(getattr(self.features, "16").branch2, "3").relu(features_16_branch2_3_bn);  features_16_branch2_3_bn = None
        features_16_branch2_4_conv = getattr(getattr(self.features, "16").branch2, "4").conv(features_16_branch2_3_relu);  features_16_branch2_3_relu = None
        features_16_branch2_4_bn = getattr(getattr(self.features, "16").branch2, "4").bn(features_16_branch2_4_conv);  features_16_branch2_4_conv = None
        features_16_branch2_4_relu = getattr(getattr(self.features, "16").branch2, "4").relu(features_16_branch2_4_bn);  features_16_branch2_4_bn = None
        return features_16_branch2_4_relu

    @ray.remote
    def b27_3(self, cat_12):
        features_16_branch3_0 = getattr(getattr(self.features, "16").branch3, "0")(cat_12);  cat_12 = None
        features_16_branch3_1_conv = getattr(getattr(self.features, "16").branch3, "1").conv(features_16_branch3_0);  features_16_branch3_0 = None
        features_16_branch3_1_bn = getattr(getattr(self.features, "16").branch3, "1").bn(features_16_branch3_1_conv);  features_16_branch3_1_conv = None
        features_16_branch3_1_relu = getattr(getattr(self.features, "16").branch3, "1").relu(features_16_branch3_1_bn);  features_16_branch3_1_bn = None
        return features_16_branch3_1_relu

    def b28(self, features_16_branch0_relu, features_16_branch1_2_relu, features_16_branch2_4_relu, features_16_branch3_1_relu):
        cat_13 = torch.cat((features_16_branch0_relu, features_16_branch1_2_relu, features_16_branch2_4_relu, features_16_branch3_1_relu), 1);  features_16_branch0_relu = features_16_branch1_2_relu = features_16_branch2_4_relu = features_16_branch3_1_relu = None
        return cat_13

    @ray.remote
    def b29_0(self, cat_13):
        features_17_branch0_conv = getattr(self.features, "17").branch0.conv(cat_13)
        features_17_branch0_bn = getattr(self.features, "17").branch0.bn(features_17_branch0_conv);  features_17_branch0_conv = None
        features_17_branch0_relu = getattr(self.features, "17").branch0.relu(features_17_branch0_bn);  features_17_branch0_bn = None
        return features_17_branch0_relu

    @ray.remote
    def b29_1(self, cat_13):
        features_17_branch1_0_conv = getattr(getattr(self.features, "17").branch1, "0").conv(cat_13)
        features_17_branch1_0_bn = getattr(getattr(self.features, "17").branch1, "0").bn(features_17_branch1_0_conv);  features_17_branch1_0_conv = None
        features_17_branch1_0_relu = getattr(getattr(self.features, "17").branch1, "0").relu(features_17_branch1_0_bn);  features_17_branch1_0_bn = None
        features_17_branch1_1_conv = getattr(getattr(self.features, "17").branch1, "1").conv(features_17_branch1_0_relu);  features_17_branch1_0_relu = None
        features_17_branch1_1_bn = getattr(getattr(self.features, "17").branch1, "1").bn(features_17_branch1_1_conv);  features_17_branch1_1_conv = None
        features_17_branch1_1_relu = getattr(getattr(self.features, "17").branch1, "1").relu(features_17_branch1_1_bn);  features_17_branch1_1_bn = None
        features_17_branch1_2_conv = getattr(getattr(self.features, "17").branch1, "2").conv(features_17_branch1_1_relu);  features_17_branch1_1_relu = None
        features_17_branch1_2_bn = getattr(getattr(self.features, "17").branch1, "2").bn(features_17_branch1_2_conv);  features_17_branch1_2_conv = None
        features_17_branch1_2_relu = getattr(getattr(self.features, "17").branch1, "2").relu(features_17_branch1_2_bn);  features_17_branch1_2_bn = None
        return features_17_branch1_2_relu

    @ray.remote
    def b29_2(self, cat_13):
        features_17_branch2_0_conv = getattr(getattr(self.features, "17").branch2, "0").conv(cat_13)
        features_17_branch2_0_bn = getattr(getattr(self.features, "17").branch2, "0").bn(features_17_branch2_0_conv);  features_17_branch2_0_conv = None
        features_17_branch2_0_relu = getattr(getattr(self.features, "17").branch2, "0").relu(features_17_branch2_0_bn);  features_17_branch2_0_bn = None
        features_17_branch2_1_conv = getattr(getattr(self.features, "17").branch2, "1").conv(features_17_branch2_0_relu);  features_17_branch2_0_relu = None
        features_17_branch2_1_bn = getattr(getattr(self.features, "17").branch2, "1").bn(features_17_branch2_1_conv);  features_17_branch2_1_conv = None
        features_17_branch2_1_relu = getattr(getattr(self.features, "17").branch2, "1").relu(features_17_branch2_1_bn);  features_17_branch2_1_bn = None
        features_17_branch2_2_conv = getattr(getattr(self.features, "17").branch2, "2").conv(features_17_branch2_1_relu);  features_17_branch2_1_relu = None
        features_17_branch2_2_bn = getattr(getattr(self.features, "17").branch2, "2").bn(features_17_branch2_2_conv);  features_17_branch2_2_conv = None
        features_17_branch2_2_relu = getattr(getattr(self.features, "17").branch2, "2").relu(features_17_branch2_2_bn);  features_17_branch2_2_bn = None
        features_17_branch2_3_conv = getattr(getattr(self.features, "17").branch2, "3").conv(features_17_branch2_2_relu);  features_17_branch2_2_relu = None
        features_17_branch2_3_bn = getattr(getattr(self.features, "17").branch2, "3").bn(features_17_branch2_3_conv);  features_17_branch2_3_conv = None
        features_17_branch2_3_relu = getattr(getattr(self.features, "17").branch2, "3").relu(features_17_branch2_3_bn);  features_17_branch2_3_bn = None
        features_17_branch2_4_conv = getattr(getattr(self.features, "17").branch2, "4").conv(features_17_branch2_3_relu);  features_17_branch2_3_relu = None
        features_17_branch2_4_bn = getattr(getattr(self.features, "17").branch2, "4").bn(features_17_branch2_4_conv);  features_17_branch2_4_conv = None
        features_17_branch2_4_relu = getattr(getattr(self.features, "17").branch2, "4").relu(features_17_branch2_4_bn);  features_17_branch2_4_bn = None
        return features_17_branch2_4_relu

    @ray.remote
    def b29_3(self, cat_13):
        features_17_branch3_0 = getattr(getattr(self.features, "17").branch3, "0")(cat_13);  cat_13 = None
        features_17_branch3_1_conv = getattr(getattr(self.features, "17").branch3, "1").conv(features_17_branch3_0);  features_17_branch3_0 = None
        features_17_branch3_1_bn = getattr(getattr(self.features, "17").branch3, "1").bn(features_17_branch3_1_conv);  features_17_branch3_1_conv = None
        features_17_branch3_1_relu = getattr(getattr(self.features, "17").branch3, "1").relu(features_17_branch3_1_bn);  features_17_branch3_1_bn = None
        return features_17_branch3_1_relu

    def b30(self, features_17_branch0_relu, features_17_branch1_2_relu, features_17_branch2_4_relu, features_17_branch3_1_relu):
        cat_14 = torch.cat((features_17_branch0_relu, features_17_branch1_2_relu, features_17_branch2_4_relu, features_17_branch3_1_relu), 1);  features_17_branch0_relu = features_17_branch1_2_relu = features_17_branch2_4_relu = features_17_branch3_1_relu = None
        return cat_14

    @ray.remote
    def b31_0(self, cat_14):
        features_18_branch0_0_conv = getattr(getattr(self.features, "18").branch0, "0").conv(cat_14)
        features_18_branch0_0_bn = getattr(getattr(self.features, "18").branch0, "0").bn(features_18_branch0_0_conv);  features_18_branch0_0_conv = None
        features_18_branch0_0_relu = getattr(getattr(self.features, "18").branch0, "0").relu(features_18_branch0_0_bn);  features_18_branch0_0_bn = None
        features_18_branch0_1_conv = getattr(getattr(self.features, "18").branch0, "1").conv(features_18_branch0_0_relu);  features_18_branch0_0_relu = None
        features_18_branch0_1_bn = getattr(getattr(self.features, "18").branch0, "1").bn(features_18_branch0_1_conv);  features_18_branch0_1_conv = None
        features_18_branch0_1_relu = getattr(getattr(self.features, "18").branch0, "1").relu(features_18_branch0_1_bn);  features_18_branch0_1_bn = None
        return features_18_branch0_1_relu

    @ray.remote
    def b31_1(self, cat_14):
        features_18_branch1_0_conv = getattr(getattr(self.features, "18").branch1, "0").conv(cat_14)
        features_18_branch1_0_bn = getattr(getattr(self.features, "18").branch1, "0").bn(features_18_branch1_0_conv);  features_18_branch1_0_conv = None
        features_18_branch1_0_relu = getattr(getattr(self.features, "18").branch1, "0").relu(features_18_branch1_0_bn);  features_18_branch1_0_bn = None
        features_18_branch1_1_conv = getattr(getattr(self.features, "18").branch1, "1").conv(features_18_branch1_0_relu);  features_18_branch1_0_relu = None
        features_18_branch1_1_bn = getattr(getattr(self.features, "18").branch1, "1").bn(features_18_branch1_1_conv);  features_18_branch1_1_conv = None
        features_18_branch1_1_relu = getattr(getattr(self.features, "18").branch1, "1").relu(features_18_branch1_1_bn);  features_18_branch1_1_bn = None
        features_18_branch1_2_conv = getattr(getattr(self.features, "18").branch1, "2").conv(features_18_branch1_1_relu);  features_18_branch1_1_relu = None
        features_18_branch1_2_bn = getattr(getattr(self.features, "18").branch1, "2").bn(features_18_branch1_2_conv);  features_18_branch1_2_conv = None
        features_18_branch1_2_relu = getattr(getattr(self.features, "18").branch1, "2").relu(features_18_branch1_2_bn);  features_18_branch1_2_bn = None
        features_18_branch1_3_conv = getattr(getattr(self.features, "18").branch1, "3").conv(features_18_branch1_2_relu);  features_18_branch1_2_relu = None
        features_18_branch1_3_bn = getattr(getattr(self.features, "18").branch1, "3").bn(features_18_branch1_3_conv);  features_18_branch1_3_conv = None
        features_18_branch1_3_relu = getattr(getattr(self.features, "18").branch1, "3").relu(features_18_branch1_3_bn);  features_18_branch1_3_bn = None
        return features_18_branch1_3_relu

    @ray.remote
    def b31_2(self, cat_14):
        features_18_branch2 = getattr(self.features, "18").branch2(cat_14);  cat_14 = None
        return features_18_branch2

    def b32(self, features_18_branch0_1_relu, features_18_branch1_3_relu, features_18_branch2):
        cat_15 = torch.cat((features_18_branch0_1_relu, features_18_branch1_3_relu, features_18_branch2), 1);  features_18_branch0_1_relu = features_18_branch1_3_relu = features_18_branch2 = None
        return cat_15

    @ray.remote
    def b33_0(self, cat_15):
        features_19_branch0_conv = getattr(self.features, "19").branch0.conv(cat_15)
        features_19_branch0_bn = getattr(self.features, "19").branch0.bn(features_19_branch0_conv);  features_19_branch0_conv = None
        features_19_branch0_relu = getattr(self.features, "19").branch0.relu(features_19_branch0_bn);  features_19_branch0_bn = None
        return features_19_branch0_relu

    @ray.remote
    def b33_1(self, x):
        def b0(cat_15):
            features_19_branch1_0_conv = getattr(self.features, "19").branch1_0.conv(cat_15)
            features_19_branch1_0_bn = getattr(self.features, "19").branch1_0.bn(features_19_branch1_0_conv);  features_19_branch1_0_conv = None
            features_19_branch1_0_relu = getattr(self.features, "19").branch1_0.relu(features_19_branch1_0_bn);  features_19_branch1_0_bn = None
            return features_19_branch1_0_relu
        def b1_0(features_19_branch1_0_relu):
            features_19_branch1_1a_conv = getattr(self.features, "19").branch1_1a.conv(features_19_branch1_0_relu)
            features_19_branch1_1a_bn = getattr(self.features, "19").branch1_1a.bn(features_19_branch1_1a_conv);  features_19_branch1_1a_conv = None
            features_19_branch1_1a_relu = getattr(self.features, "19").branch1_1a.relu(features_19_branch1_1a_bn);  features_19_branch1_1a_bn = None
            return features_19_branch1_1a_relu
        def b1_1(features_19_branch1_0_relu):
            features_19_branch1_1b_conv = getattr(self.features, "19").branch1_1b.conv(features_19_branch1_0_relu);  features_19_branch1_0_relu = None
            features_19_branch1_1b_bn = getattr(self.features, "19").branch1_1b.bn(features_19_branch1_1b_conv);  features_19_branch1_1b_conv = None
            features_19_branch1_1b_relu = getattr(self.features, "19").branch1_1b.relu(features_19_branch1_1b_bn);  features_19_branch1_1b_bn = None
            return features_19_branch1_1b_relu
        def b2(features_19_branch1_1a_relu, features_19_branch1_1b_relu):
            cat_16 = torch.cat((features_19_branch1_1a_relu, features_19_branch1_1b_relu), 1);  features_19_branch1_1a_relu = features_19_branch1_1b_relu = None
            return cat_16
        def forward(x):
            x = b0(x)
            _b1_0 = b1_0(x)
            _b1_1 = b1_1(x)
            x = b2(_b1_0,_b1_1)
            return x
        return forward(x)
    @ray.remote
    def b33_2(self, x):
        def b0(cat_15):
            features_19_branch2_0_conv = getattr(self.features, "19").branch2_0.conv(cat_15)
            features_19_branch2_0_bn = getattr(self.features, "19").branch2_0.bn(features_19_branch2_0_conv);  features_19_branch2_0_conv = None
            features_19_branch2_0_relu = getattr(self.features, "19").branch2_0.relu(features_19_branch2_0_bn);  features_19_branch2_0_bn = None
            features_19_branch2_1_conv = getattr(self.features, "19").branch2_1.conv(features_19_branch2_0_relu);  features_19_branch2_0_relu = None
            features_19_branch2_1_bn = getattr(self.features, "19").branch2_1.bn(features_19_branch2_1_conv);  features_19_branch2_1_conv = None
            features_19_branch2_1_relu = getattr(self.features, "19").branch2_1.relu(features_19_branch2_1_bn);  features_19_branch2_1_bn = None
            features_19_branch2_2_conv = getattr(self.features, "19").branch2_2.conv(features_19_branch2_1_relu);  features_19_branch2_1_relu = None
            features_19_branch2_2_bn = getattr(self.features, "19").branch2_2.bn(features_19_branch2_2_conv);  features_19_branch2_2_conv = None
            features_19_branch2_2_relu = getattr(self.features, "19").branch2_2.relu(features_19_branch2_2_bn);  features_19_branch2_2_bn = None
            return features_19_branch2_2_relu
        def b1_0(features_19_branch2_2_relu):
            features_19_branch2_3a_conv = getattr(self.features, "19").branch2_3a.conv(features_19_branch2_2_relu)
            features_19_branch2_3a_bn = getattr(self.features, "19").branch2_3a.bn(features_19_branch2_3a_conv);  features_19_branch2_3a_conv = None
            features_19_branch2_3a_relu = getattr(self.features, "19").branch2_3a.relu(features_19_branch2_3a_bn);  features_19_branch2_3a_bn = None
            return features_19_branch2_3a_relu
        def b1_1(features_19_branch2_2_relu):
            features_19_branch2_3b_conv = getattr(self.features, "19").branch2_3b.conv(features_19_branch2_2_relu);  features_19_branch2_2_relu = None
            features_19_branch2_3b_bn = getattr(self.features, "19").branch2_3b.bn(features_19_branch2_3b_conv);  features_19_branch2_3b_conv = None
            features_19_branch2_3b_relu = getattr(self.features, "19").branch2_3b.relu(features_19_branch2_3b_bn);  features_19_branch2_3b_bn = None
            return features_19_branch2_3b_relu
        def b2(features_19_branch2_3a_relu, features_19_branch2_3b_relu):
            cat_17 = torch.cat((features_19_branch2_3a_relu, features_19_branch2_3b_relu), 1);  features_19_branch2_3a_relu = features_19_branch2_3b_relu = None
            return cat_17
        def forward(x):
            x = b0(x)
            _b1_0 = b1_0(x)
            _b1_1 = b1_1(x)
            x = b2(_b1_0,_b1_1)
            return x
        return forward(x)
    @ray.remote
    def b33_3(self, cat_15):
        features_19_branch3_0 = getattr(getattr(self.features, "19").branch3, "0")(cat_15);  cat_15 = None
        features_19_branch3_1_conv = getattr(getattr(self.features, "19").branch3, "1").conv(features_19_branch3_0);  features_19_branch3_0 = None
        features_19_branch3_1_bn = getattr(getattr(self.features, "19").branch3, "1").bn(features_19_branch3_1_conv);  features_19_branch3_1_conv = None
        features_19_branch3_1_relu = getattr(getattr(self.features, "19").branch3, "1").relu(features_19_branch3_1_bn);  features_19_branch3_1_bn = None
        return features_19_branch3_1_relu

    def b34(self, features_19_branch0_relu, cat_16, cat_17, features_19_branch3_1_relu):
        cat_18 = torch.cat((features_19_branch0_relu, cat_16, cat_17, features_19_branch3_1_relu), 1);  features_19_branch0_relu = cat_16 = cat_17 = features_19_branch3_1_relu = None
        return cat_18

    @ray.remote
    def b35_0(self, cat_18):
        features_20_branch0_conv = getattr(self.features, "20").branch0.conv(cat_18)
        features_20_branch0_bn = getattr(self.features, "20").branch0.bn(features_20_branch0_conv);  features_20_branch0_conv = None
        features_20_branch0_relu = getattr(self.features, "20").branch0.relu(features_20_branch0_bn);  features_20_branch0_bn = None
        return features_20_branch0_relu

    @ray.remote
    def b35_1(self, x):
        def b0(cat_18):
            features_20_branch1_0_conv = getattr(self.features, "20").branch1_0.conv(cat_18)
            features_20_branch1_0_bn = getattr(self.features, "20").branch1_0.bn(features_20_branch1_0_conv);  features_20_branch1_0_conv = None
            features_20_branch1_0_relu = getattr(self.features, "20").branch1_0.relu(features_20_branch1_0_bn);  features_20_branch1_0_bn = None
            return features_20_branch1_0_relu
        def b1_0(features_20_branch1_0_relu):
            features_20_branch1_1a_conv = getattr(self.features, "20").branch1_1a.conv(features_20_branch1_0_relu)
            features_20_branch1_1a_bn = getattr(self.features, "20").branch1_1a.bn(features_20_branch1_1a_conv);  features_20_branch1_1a_conv = None
            features_20_branch1_1a_relu = getattr(self.features, "20").branch1_1a.relu(features_20_branch1_1a_bn);  features_20_branch1_1a_bn = None
            return features_20_branch1_1a_relu
        def b1_1(features_20_branch1_0_relu):
            features_20_branch1_1b_conv = getattr(self.features, "20").branch1_1b.conv(features_20_branch1_0_relu);  features_20_branch1_0_relu = None
            features_20_branch1_1b_bn = getattr(self.features, "20").branch1_1b.bn(features_20_branch1_1b_conv);  features_20_branch1_1b_conv = None
            features_20_branch1_1b_relu = getattr(self.features, "20").branch1_1b.relu(features_20_branch1_1b_bn);  features_20_branch1_1b_bn = None
            return features_20_branch1_1b_relu
        def b2(features_20_branch1_1a_relu, features_20_branch1_1b_relu):
            cat_19 = torch.cat((features_20_branch1_1a_relu, features_20_branch1_1b_relu), 1);  features_20_branch1_1a_relu = features_20_branch1_1b_relu = None
            return cat_19
        def forward(x):
            x = b0(x)
            _b1_0 = b1_0(x)
            _b1_1 = b1_1(x)
            x = b2(_b1_0,_b1_1)
            return x
        return forward(x)
    @ray.remote
    def b35_2(self, x):
        def b0(cat_18):
            features_20_branch2_0_conv = getattr(self.features, "20").branch2_0.conv(cat_18)
            features_20_branch2_0_bn = getattr(self.features, "20").branch2_0.bn(features_20_branch2_0_conv);  features_20_branch2_0_conv = None
            features_20_branch2_0_relu = getattr(self.features, "20").branch2_0.relu(features_20_branch2_0_bn);  features_20_branch2_0_bn = None
            features_20_branch2_1_conv = getattr(self.features, "20").branch2_1.conv(features_20_branch2_0_relu);  features_20_branch2_0_relu = None
            features_20_branch2_1_bn = getattr(self.features, "20").branch2_1.bn(features_20_branch2_1_conv);  features_20_branch2_1_conv = None
            features_20_branch2_1_relu = getattr(self.features, "20").branch2_1.relu(features_20_branch2_1_bn);  features_20_branch2_1_bn = None
            features_20_branch2_2_conv = getattr(self.features, "20").branch2_2.conv(features_20_branch2_1_relu);  features_20_branch2_1_relu = None
            features_20_branch2_2_bn = getattr(self.features, "20").branch2_2.bn(features_20_branch2_2_conv);  features_20_branch2_2_conv = None
            features_20_branch2_2_relu = getattr(self.features, "20").branch2_2.relu(features_20_branch2_2_bn);  features_20_branch2_2_bn = None
            return features_20_branch2_2_relu
        def b1_0(features_20_branch2_2_relu):
            features_20_branch2_3a_conv = getattr(self.features, "20").branch2_3a.conv(features_20_branch2_2_relu)
            features_20_branch2_3a_bn = getattr(self.features, "20").branch2_3a.bn(features_20_branch2_3a_conv);  features_20_branch2_3a_conv = None
            features_20_branch2_3a_relu = getattr(self.features, "20").branch2_3a.relu(features_20_branch2_3a_bn);  features_20_branch2_3a_bn = None
            return features_20_branch2_3a_relu
        def b1_1(features_20_branch2_2_relu):
            features_20_branch2_3b_conv = getattr(self.features, "20").branch2_3b.conv(features_20_branch2_2_relu);  features_20_branch2_2_relu = None
            features_20_branch2_3b_bn = getattr(self.features, "20").branch2_3b.bn(features_20_branch2_3b_conv);  features_20_branch2_3b_conv = None
            features_20_branch2_3b_relu = getattr(self.features, "20").branch2_3b.relu(features_20_branch2_3b_bn);  features_20_branch2_3b_bn = None
            return features_20_branch2_3b_relu
        def b2(features_20_branch2_3a_relu, features_20_branch2_3b_relu):
            cat_20 = torch.cat((features_20_branch2_3a_relu, features_20_branch2_3b_relu), 1);  features_20_branch2_3a_relu = features_20_branch2_3b_relu = None
            return cat_20
        def forward(x):
            x = b0(x)
            _b1_0 = b1_0(x)
            _b1_1 = b1_1(x)
            x = b2(_b1_0,_b1_1)
            return x
        return forward(x)
    @ray.remote
    def b35_3(self, cat_18):
        features_20_branch3_0 = getattr(getattr(self.features, "20").branch3, "0")(cat_18);  cat_18 = None
        features_20_branch3_1_conv = getattr(getattr(self.features, "20").branch3, "1").conv(features_20_branch3_0);  features_20_branch3_0 = None
        features_20_branch3_1_bn = getattr(getattr(self.features, "20").branch3, "1").bn(features_20_branch3_1_conv);  features_20_branch3_1_conv = None
        features_20_branch3_1_relu = getattr(getattr(self.features, "20").branch3, "1").relu(features_20_branch3_1_bn);  features_20_branch3_1_bn = None
        return features_20_branch3_1_relu

    def b36(self, features_20_branch0_relu, cat_19, cat_20, features_20_branch3_1_relu):
        cat_21 = torch.cat((features_20_branch0_relu, cat_19, cat_20, features_20_branch3_1_relu), 1);  features_20_branch0_relu = cat_19 = cat_20 = features_20_branch3_1_relu = None
        return cat_21

    @ray.remote
    def b37_0(self, cat_21):
        features_21_branch0_conv = getattr(self.features, "21").branch0.conv(cat_21)
        features_21_branch0_bn = getattr(self.features, "21").branch0.bn(features_21_branch0_conv);  features_21_branch0_conv = None
        features_21_branch0_relu = getattr(self.features, "21").branch0.relu(features_21_branch0_bn);  features_21_branch0_bn = None
        return features_21_branch0_relu

    @ray.remote
    def b37_1(self, x):
        def b0(cat_21):
            features_21_branch1_0_conv = getattr(self.features, "21").branch1_0.conv(cat_21)
            features_21_branch1_0_bn = getattr(self.features, "21").branch1_0.bn(features_21_branch1_0_conv);  features_21_branch1_0_conv = None
            features_21_branch1_0_relu = getattr(self.features, "21").branch1_0.relu(features_21_branch1_0_bn);  features_21_branch1_0_bn = None
            return features_21_branch1_0_relu
        def b1_0(features_21_branch1_0_relu):
            features_21_branch1_1a_conv = getattr(self.features, "21").branch1_1a.conv(features_21_branch1_0_relu)
            features_21_branch1_1a_bn = getattr(self.features, "21").branch1_1a.bn(features_21_branch1_1a_conv);  features_21_branch1_1a_conv = None
            features_21_branch1_1a_relu = getattr(self.features, "21").branch1_1a.relu(features_21_branch1_1a_bn);  features_21_branch1_1a_bn = None
            return features_21_branch1_1a_relu
        def b1_1(features_21_branch1_0_relu):
            features_21_branch1_1b_conv = getattr(self.features, "21").branch1_1b.conv(features_21_branch1_0_relu);  features_21_branch1_0_relu = None
            features_21_branch1_1b_bn = getattr(self.features, "21").branch1_1b.bn(features_21_branch1_1b_conv);  features_21_branch1_1b_conv = None
            features_21_branch1_1b_relu = getattr(self.features, "21").branch1_1b.relu(features_21_branch1_1b_bn);  features_21_branch1_1b_bn = None
            return features_21_branch1_1b_relu
        def b2(features_21_branch1_1a_relu, features_21_branch1_1b_relu):
            cat_22 = torch.cat((features_21_branch1_1a_relu, features_21_branch1_1b_relu), 1);  features_21_branch1_1a_relu = features_21_branch1_1b_relu = None
            return cat_22
        def forward(x):
            x = b0(x)
            _b1_0 = b1_0(x)
            _b1_1 = b1_1(x)
            x = b2(_b1_0,_b1_1)
            return x
        return forward(x)
    @ray.remote
    def b37_2(self, x):
        def b0(cat_21):
            features_21_branch2_0_conv = getattr(self.features, "21").branch2_0.conv(cat_21)
            features_21_branch2_0_bn = getattr(self.features, "21").branch2_0.bn(features_21_branch2_0_conv);  features_21_branch2_0_conv = None
            features_21_branch2_0_relu = getattr(self.features, "21").branch2_0.relu(features_21_branch2_0_bn);  features_21_branch2_0_bn = None
            features_21_branch2_1_conv = getattr(self.features, "21").branch2_1.conv(features_21_branch2_0_relu);  features_21_branch2_0_relu = None
            features_21_branch2_1_bn = getattr(self.features, "21").branch2_1.bn(features_21_branch2_1_conv);  features_21_branch2_1_conv = None
            features_21_branch2_1_relu = getattr(self.features, "21").branch2_1.relu(features_21_branch2_1_bn);  features_21_branch2_1_bn = None
            features_21_branch2_2_conv = getattr(self.features, "21").branch2_2.conv(features_21_branch2_1_relu);  features_21_branch2_1_relu = None
            features_21_branch2_2_bn = getattr(self.features, "21").branch2_2.bn(features_21_branch2_2_conv);  features_21_branch2_2_conv = None
            features_21_branch2_2_relu = getattr(self.features, "21").branch2_2.relu(features_21_branch2_2_bn);  features_21_branch2_2_bn = None
            return features_21_branch2_2_relu
        def b1_0(features_21_branch2_2_relu):
            features_21_branch2_3a_conv = getattr(self.features, "21").branch2_3a.conv(features_21_branch2_2_relu)
            features_21_branch2_3a_bn = getattr(self.features, "21").branch2_3a.bn(features_21_branch2_3a_conv);  features_21_branch2_3a_conv = None
            features_21_branch2_3a_relu = getattr(self.features, "21").branch2_3a.relu(features_21_branch2_3a_bn);  features_21_branch2_3a_bn = None
            return features_21_branch2_3a_relu
        def b1_1(features_21_branch2_2_relu):
            features_21_branch2_3b_conv = getattr(self.features, "21").branch2_3b.conv(features_21_branch2_2_relu);  features_21_branch2_2_relu = None
            features_21_branch2_3b_bn = getattr(self.features, "21").branch2_3b.bn(features_21_branch2_3b_conv);  features_21_branch2_3b_conv = None
            features_21_branch2_3b_relu = getattr(self.features, "21").branch2_3b.relu(features_21_branch2_3b_bn);  features_21_branch2_3b_bn = None
            return features_21_branch2_3b_relu
        def b2(features_21_branch2_3a_relu, features_21_branch2_3b_relu):
            cat_23 = torch.cat((features_21_branch2_3a_relu, features_21_branch2_3b_relu), 1);  features_21_branch2_3a_relu = features_21_branch2_3b_relu = None
            return cat_23
        def forward(x):
            x = b0(x)
            _b1_0 = b1_0(x)
            _b1_1 = b1_1(x)
            x = b2(_b1_0,_b1_1)
            return x
        return forward(x)
    @ray.remote
    def b37_3(self, cat_21):
        features_21_branch3_0 = getattr(getattr(self.features, "21").branch3, "0")(cat_21);  cat_21 = None
        features_21_branch3_1_conv = getattr(getattr(self.features, "21").branch3, "1").conv(features_21_branch3_0);  features_21_branch3_0 = None
        features_21_branch3_1_bn = getattr(getattr(self.features, "21").branch3, "1").bn(features_21_branch3_1_conv);  features_21_branch3_1_conv = None
        features_21_branch3_1_relu = getattr(getattr(self.features, "21").branch3, "1").relu(features_21_branch3_1_bn);  features_21_branch3_1_bn = None
        return features_21_branch3_1_relu

    def b38(self, features_21_branch0_relu, cat_22, cat_23, features_21_branch3_1_relu):
        cat_24 = torch.cat((features_21_branch0_relu, cat_22, cat_23, features_21_branch3_1_relu), 1);  features_21_branch0_relu = cat_22 = cat_23 = features_21_branch3_1_relu = None
        return cat_24

    def tail(self, cat_24, kernel_size = 8):
        avg_pool2d = torch._C._nn.avg_pool2d(cat_24, kernel_size = 8);  cat_24 = None
        size = avg_pool2d.size(0)
        view = avg_pool2d.view(size, -1);  avg_pool2d = size = None
        last_linear = self.last_linear(view);  view = None
        return last_linear

    def forward(self, x):
        x = self.b0(x)
        b1_0 = self.b1_0.remote(self, x)
        b1_1 = self.b1_1.remote(self, x)
        x = self.b2(ray.get(b1_0), ray.get(b1_1))
        b3_0 = self.b3_0.remote(self, x)
        b3_1 = self.b3_1.remote(self, x)
        x = self.b4(ray.get(b3_0), ray.get(b3_1))
        b5_0 = self.b5_0.remote(self, x)
        b5_1 = self.b5_1.remote(self, x)
        x = self.b6(ray.get(b5_0), ray.get(b5_1))
        b7_0 = self.b7_0.remote(self, x)
        b7_1 = self.b7_1.remote(self, x)
        b7_2 = self.b7_2.remote(self, x)
        b7_3 = self.b7_3.remote(self, x)
        x = self.b8(ray.get(b7_0), ray.get(b7_1), ray.get(b7_2), ray.get(b7_3))
        b9_0 = self.b9_0.remote(self, x)
        b9_1 = self.b9_1.remote(self, x)
        b9_2 = self.b9_2.remote(self, x)
        b9_3 = self.b9_3.remote(self, x)
        x = self.b10(ray.get(b9_0), ray.get(b9_1), ray.get(b9_2), ray.get(b9_3))
        b11_0 = self.b11_0.remote(self, x)
        b11_1 = self.b11_1.remote(self, x)
        b11_2 = self.b11_2.remote(self, x)
        b11_3 = self.b11_3.remote(self, x)
        x = self.b12(ray.get(b11_0), ray.get(b11_1), ray.get(b11_2), ray.get(b11_3))
        b13_0 = self.b13_0.remote(self, x)
        b13_1 = self.b13_1.remote(self, x)
        b13_2 = self.b13_2.remote(self, x)
        b13_3 = self.b13_3.remote(self, x)
        x = self.b14(ray.get(b13_0), ray.get(b13_1), ray.get(b13_2), ray.get(b13_3))
        b15_0 = self.b15_0.remote(self, x)
        b15_1 = self.b15_1.remote(self, x)
        b15_2 = self.b15_2.remote(self, x)
        x = self.b16(ray.get(b15_0), ray.get(b15_1), ray.get(b15_2))
        b17_0 = self.b17_0.remote(self, x)
        b17_1 = self.b17_1.remote(self, x)
        b17_2 = self.b17_2.remote(self, x)
        b17_3 = self.b17_3.remote(self, x)
        x = self.b18(ray.get(b17_0), ray.get(b17_1), ray.get(b17_2), ray.get(b17_3))
        b19_0 = self.b19_0.remote(self, x)
        b19_1 = self.b19_1.remote(self, x)
        b19_2 = self.b19_2.remote(self, x)
        b19_3 = self.b19_3.remote(self, x)
        x = self.b20(ray.get(b19_0), ray.get(b19_1), ray.get(b19_2), ray.get(b19_3))
        b21_0 = self.b21_0.remote(self, x)
        b21_1 = self.b21_1.remote(self, x)
        b21_2 = self.b21_2.remote(self, x)
        b21_3 = self.b21_3.remote(self, x)
        x = self.b22(ray.get(b21_0), ray.get(b21_1), ray.get(b21_2), ray.get(b21_3))
        b23_0 = self.b23_0.remote(self, x)
        b23_1 = self.b23_1.remote(self, x)
        b23_2 = self.b23_2.remote(self, x)
        b23_3 = self.b23_3.remote(self, x)
        x = self.b24(ray.get(b23_0), ray.get(b23_1), ray.get(b23_2), ray.get(b23_3))
        b25_0 = self.b25_0.remote(self, x)
        b25_1 = self.b25_1.remote(self, x)
        b25_2 = self.b25_2.remote(self, x)
        b25_3 = self.b25_3.remote(self, x)
        x = self.b26(ray.get(b25_0), ray.get(b25_1), ray.get(b25_2), ray.get(b25_3))
        b27_0 = self.b27_0.remote(self, x)
        b27_1 = self.b27_1.remote(self, x)
        b27_2 = self.b27_2.remote(self, x)
        b27_3 = self.b27_3.remote(self, x)
        x = self.b28(ray.get(b27_0), ray.get(b27_1), ray.get(b27_2), ray.get(b27_3))
        b29_0 = self.b29_0.remote(self, x)
        b29_1 = self.b29_1.remote(self, x)
        b29_2 = self.b29_2.remote(self, x)
        b29_3 = self.b29_3.remote(self, x)
        x = self.b30(ray.get(b29_0), ray.get(b29_1), ray.get(b29_2), ray.get(b29_3))
        b31_0 = self.b31_0.remote(self, x)
        b31_1 = self.b31_1.remote(self, x)
        b31_2 = self.b31_2.remote(self, x)
        x = self.b32(ray.get(b31_0), ray.get(b31_1), ray.get(b31_2))
        b33_0 = self.b33_0.remote(self, x)
        b33_1 = self.b33_1.remote(self, x)
        b33_2 = self.b33_2.remote(self, x)
        b33_3 = self.b33_3.remote(self, x)
        x = self.b34(ray.get(b33_0), ray.get(b33_1), ray.get(b33_2), ray.get(b33_3))
        b35_0 = self.b35_0.remote(self, x)
        b35_1 = self.b35_1.remote(self, x)
        b35_2 = self.b35_2.remote(self, x)
        b35_3 = self.b35_3.remote(self, x)
        x = self.b36(ray.get(b35_0), ray.get(b35_1), ray.get(b35_2), ray.get(b35_3))
        b37_0 = self.b37_0.remote(self, x)
        b37_1 = self.b37_1.remote(self, x)
        b37_2 = self.b37_2.remote(self, x)
        b37_3 = self.b37_3.remote(self, x)
        x = self.b38(ray.get(b37_0), ray.get(b37_1), ray.get(b37_2), ray.get(b37_3))
        x = self.tail(x)
        return x
