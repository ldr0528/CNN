import math
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.coatnet import coatnet_0
from models.resnet18 import resnet18
from models.vgg16 import vgg16
from models.efficientnet import efficientnetb0
from dataloader import CustomDataset
from models.densenet121 import densenet121
from models.mobilenetv3 import mobilenetv3

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 定义数据预处理的转换
transform = transforms.Compose([
     #RandAugment(n=2, m=10),
    # transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
])
# Example usage:
root_dir = 'HOG_2'

epochs = 10
batch_size = 8
lr = 0.00005
lrf = 0.01
num_classes = 3
num_workers = 0

# 定义训练数据集
# train_dataset = ImageFolder(root='../dataset/STFT_HOG_OSARship/train', transform=transform)
#
# # 定义测试数据集
# test_dataset = ImageFolder(root='../dataset/STFT_HOG_OSARship/val', transform=transform)

# mixup_train_dataset = MixUpDataset(train_dataset, alpha=alpha)
#
# mixup_test_dataset = MixUpDataset(train_dataset, alpha=alpha)

# 定义训练数据集的dataloader
# train_dataloader = DataLoader(mixup_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#
# # 定义测试数据集的dataloader
# test_dataloader = DataLoader(mixup_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#
# # 定义测试数据集的dataloader
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

train_dataset = CustomDataset(root_dir, split='Train', transform=transform)
test_dataset = CustomDataset(root_dir, split='Test', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = coatnet_0(num_classes=num_classes)
#model = efficientnetb0(num_classes=num_classes)
#model = resnet18(num_classes=num_classes)
#model = vgg16(num_classes=num_classes)
#model = densenet121(num_classes=num_classes)
model = mobilenetv3(num_classes=num_classes)

model = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
# lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)



optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)     #优化器 (optimizer)：SGD（带动量）、RMSprop
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00005)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)     #学习率调度器 (scheduler)：StepLR、ReduceLROnPlateau
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,verbose=True)



criterion = torch.nn.CrossEntropyLoss()     #损失函数 (criterion)

start_epoch = 0
checkpoint_resume = False
if checkpoint_resume:  # 恢复上次训练模型
    checkpoint = torch.load("checkpoint/lastmodel.pkl")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.cuda()
    start_epoch = checkpoint['epoch']
    scheduler.last_epoch = start_epoch
    optimizer.load_state_dict(checkpoint['opt'])

# 训练模型
best_acc = 0
for epoch in range(start_epoch, epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0  # 初始化训练损失为0.0
    tqdm_batch = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
    for inputs, labels in tqdm_batch:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 清除梯度
        optimizer.zero_grad()
        # 前向传播

        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)

        loss.backward()  # 对损失进行反向传播，计算梯度
        optimizer.step()  # 使用优化器更新模型参数
        train_loss += loss.item() * inputs.size(0)  # 累加批次损失值
    train_loss /= len(train_dataset)  # 计算平均训练损失值
    print(f"第{epoch + 1}轮的训练损失为{train_loss:.4f}")

    model.eval()  # 设置模型为评估模式
    test_loss = 0.0  # 初始化测试损失为0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度值，节省内存空间和计算时间
        for inputs, labels in tqdm(test_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)  # 前向传播，得到预测值
            loss = criterion(outputs, labels.long())  # 计算损失值
            test_loss += loss.item() * inputs.size(0)  # 累加批次损失值

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_dataset)  # 计算平均测试损失值
    accuracy = correct / total
    print(f"第{epoch + 1}轮的测试损失为{test_loss:.4f}, 准确率为{accuracy}")



    scheduler.step()  # 更新学习率

    checkpoint = {"model_state_dict": model.state_dict(),
                  "opt": optimizer.state_dict(),
                  "acc": accuracy,
                  "epoch": epoch}
    if accuracy > best_acc:
        # path_checkpoint = "./checkpoint/checkpoint_{}_epoch.pkl".format(epoch)  # 定义保存路径，包含epoch编号
        path_checkpoint = "checkpoint/mobilenet.pkl".format(epoch)  # 定义保存路径，包含epoch编号
        torch.save(checkpoint, path_checkpoint)  # 保存字典到文件中

        print("Best model saved!")
        # 更新最佳指标
        best_acc = accuracy
    print(f"最高准确率为{best_acc}")
    print("\n")
    path_checkpoint = "checkpoint/lastmodel.pkl".format(epoch)  # 定义保存路径，包含epoch编号
    torch.save(checkpoint, path_checkpoint)  # 保存字典到文件中

