import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道1，输出通道32，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道32，输出通道64，卷积核大小3x3
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层，输入特征数为64*7*7，输出特征数128
        self.fc2 = nn.Linear(128, 10)  # 全连接层，输入特征数128，输出特征数10（MNIST数字类别）

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一个卷积层后接ReLU激活函数
        x = F.max_pool2d(x, 2)  # 最大池化，池化窗口大小2x2
        x = F.relu(self.conv2(x))  # 第二个卷积层后接ReLU激活函数
        x = F.max_pool2d(x, 2)  # 最大池化，池化窗口大小2x2
        x = x.view(-1, 64 * 7 * 7)  # 展平特征图
        x = F.relu(self.fc1(x))  # 第一个全连接层后接ReLU激活函数
        x = self.fc2(x)  # 第二个全连接层
        return F.log_softmax(x, dim=1)  # 应用log_softmax函数


# 实例化模型
model = CNN()
print(model)
