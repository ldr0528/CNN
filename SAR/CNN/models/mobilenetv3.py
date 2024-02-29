import torch
from torchvision import models
from torch import nn


def mobilenetv3(num_classes, pretrained=True):
    model_ft = models.mobilenet_v3_large(pretrained=pretrained)
    # 替换第一层卷积，将输入通道从3改为1
    model_ft.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    num_ftrs = model_ft.classifier[0].out_features
    model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
    return model_ft
