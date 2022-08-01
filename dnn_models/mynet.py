import torch
import torch.nn as nn

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=4, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=6, stride=3, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        y1 = self.conv2(x)
        y1 = self.relu(y1)
        y2 = self.conv2(x)
        y = y1 + y2
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.pool1(x)

        return y


if __name__ == "__main__":
    net = MyNet()
    net.eval()

    input = torch.randn(1, 3, 224, 224)
    output = net(input)
    print(output)