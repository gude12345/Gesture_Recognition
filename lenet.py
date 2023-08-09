from torch import nn
from torch.nn import functional as F


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        #第一个容器
        self.conv_seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        #第二个容器
        self.fc_seq = nn.Sequential(
            nn.Linear(32*6*6, 256),
            nn.ReLU(),
            nn.Linear(256, 10)       #全连接层，第二项为分类个数
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_seq(x)
        x = x.reshape(batch_size, 32*6*6)
        out = self.fc_seq(x)
        return out


class Lenet_i(nn.Module):
    def __init__(self):
        super(Lenet_i, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32*6*6, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = x.reshape(batch_size, 32*6*6)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out
