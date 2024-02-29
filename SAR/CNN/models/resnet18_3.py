from torchvision import models
from torch import nn
import sys


def resnet18_3(num_classes):
    model_ft = models.resnet18(pretrained=True)
    # 替换第一层卷积，将输入通道从3改为1
    # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft



