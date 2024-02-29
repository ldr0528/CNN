from torchvision import models
import torch.nn as nn

def efficientnetb0(num_classes=3):
    # 加载预训练的 EfficientNet-B0 模型
    model_ft = models.efficientnet_b0(pretrained=True)

    # 替换第一层卷积，将输入通道从3改为1
    model_ft.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    # 替换分类器的输出层以适应新的类别数
    model_ft.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    return model_ft
