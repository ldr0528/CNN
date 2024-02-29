import torch
from torchvision import models
from torch import nn

class vgg16(nn.Module):
    def __init__(self, num_classes):
        super(vgg16, self).__init__()
        # 直接加载预训练的 VGG16 模型
        vgg16_net = models.vgg16_bn(pretrained=True)

        # 替换第一层卷积，将输入通道从3改为1
        vgg16_net.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        self.features = vgg16_net.features
        self.avgpool = vgg16_net.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class vgg19(nn.Module):
    def __init__(self, num_classes):
        super(vgg19, self).__init__()
        # 直接加载预训练的 VGG19 模型
        vgg19_net = models.vgg19_bn(pretrained=True)

        self.features = vgg19_net.features
        self.avgpool = vgg19_net.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
