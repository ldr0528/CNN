import torch
from torchvision import models
from torch import nn

class densenet121(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(densenet121, self).__init__()

        # 直接在创建模型时加载预训练权重
        self.net = models.densenet121(pretrained=pretrained)

        # 替换第一层卷积，将输入通道从3改为1
        self.net.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 替换分类器层，以适应新的类别数
        self.net.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        y = self.net(x)
        return y
