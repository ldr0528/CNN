import itertools
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from numpy import interp
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

# from models.coatnet import coatnet_0
from models.resnet18 import resnet18
from models.vgg16 import vgg16
from models.densenet121 import densenet121
from models.mobilenetv3 import mobilenetv3
from models.efficientnet import efficientnetb0

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 定义数据预处理的转换
transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
])

batch_size = 8
num_classes = 3

num_workers = 0

# 定义测试数据集
#test_dataset = ImageFolder(root='D:\dataset\Feature\CWT+HOG_2', transform=transform)
root_dir = 'SSTN_2'

# 定义测试数据集的dataloader
test_dataset = CustomDataset(root_dir, split='Test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = coatnet_0(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()

models_name = [
    'resnet-18+2nd-fsst+hog',
    'vgg-16',
    'densenet-121',
    'mobilenet-v3',
    'efficientnet-b0',
    # 'resnet+stft+hog',
    # 'resnet+hog',
]

#simplified_names = ['+'.join(name.split('+')[:3]) for name in models_name]


models = []
for name in models_name:
    model = None
    # 提取模型名称的主要部分
    model_name = name.split('+')[0]

    checkpoint = torch.load(f"checkpoint1/{name}.pkl")
    if model_name == 'resnet-18':
        model = resnet18(num_classes=num_classes)
    elif model_name == 'vgg-16':
        model = vgg16(num_classes=num_classes)
    elif model_name == 'densenet-121':
        model = densenet121(num_classes=num_classes)
    elif model_name == 'mobilenet-v3':
        model = mobilenetv3(num_classes=num_classes)
    elif model_name == 'efficientnet-b0':
        model = efficientnetb0(num_classes=num_classes)

    if model:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.cuda()  # 如果您使用的是GPU，则取消注释这一行
        model.eval()
        models.append(model)
    else:
        print(f"未找到模型：{model_name}")


# 初始化测试损失为0.0
test_loss = 0.0
correct = 0
total = 0

# 初始化混淆矩阵
all_labels = []
all_predicted_probs = []
conf_matrix = np.zeros((num_classes, num_classes))
model_results = []

model_styles = [
    {'color': (69 / 255.0, 189 / 255.0, 155 / 255.0), 'linestyle': '-'},
    {'color': (43 / 255.0, 85 / 255.0, 125 / 255.0), 'linestyle': '--'},
    {'color': (240 / 255.0, 81 / 255.0, 121 / 255.0), 'linestyle': '-.'},
    {'color': (250 / 255.0, 192 / 255.0, 15 / 255.0), 'linestyle': ':'},
    {'color': (30 / 255.0, 144 / 255.0, 255 / 255.0), 'linestyle': '-'}
]
# 循环遍历每个模型
for i, model in enumerate(models):
    with torch.no_grad():  # 不计算梯度值，节省内存空间和计算时间
        all_labels = []
        all_predicted_probs = []

        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)  # 前向传播，得到预测值
            loss = criterion(outputs, labels)  # 计算损失值
            test_loss += loss.item()  # 累加批次损失值

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集预测概率用于计算ROC曲线
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predicted_probs.extend(probs.cpu().numpy())

        # 添加绘制ROC曲线和计算F1-score的代码
        prediction = np.array(all_predicted_probs)
        labels = torch.tensor(all_labels)
        labels = labels.reshape((labels.shape[0], 1))
        labels_onehot = torch.zeros(labels.shape[0], num_classes)
        labels_onehot.scatter_(dim=1, index=labels, value=1)
        labels_onehot = np.array(labels_onehot)
        # 计算每个模型的 ROC 曲线和 AUC
        fpr_model = dict()
        tpr_model = dict()
        roc_auc_model = dict()

        for class_idx in range(num_classes):
            fpr_model[class_idx], tpr_model[class_idx], _ = roc_curve(labels_onehot[:, class_idx], prediction[:, class_idx])
            roc_auc_model[class_idx] = auc(fpr_model[class_idx], tpr_model[class_idx])

        fpr_model["micro"], tpr_model["micro"], _ = roc_curve(labels_onehot.ravel(), prediction.ravel())
        roc_auc_model["micro"] = auc(fpr_model["micro"], tpr_model["micro"])

        all_fpr_model = np.unique(np.concatenate([fpr_model[class_idx] for class_idx in range(num_classes)]))
        mean_tpr_model = np.zeros_like(all_fpr_model)

        for class_idx in range(num_classes):
            mean_tpr_model += interp(all_fpr_model, fpr_model[class_idx], tpr_model[class_idx])

        mean_tpr_model /= num_classes
        fpr_model["macro"] = all_fpr_model
        tpr_model["macro"] = mean_tpr_model
        roc_auc_model["macro"] = auc(fpr_model["macro"], tpr_model["macro"])

        model_results.append((fpr_model, tpr_model, roc_auc_model, model_styles[i]))


# ----------------------------------------
# 绘制 ROC 曲线
# ----------------------------------------
plt.figure()

lw = 2
colors = itertools.cycle([(69 / 255.0, 189 / 255.0, 155 / 255.0), (43 / 255.0, 85 / 255.0, 125 / 255.0),
                          (240 / 255.0, 81 / 255.0, 121 / 255.0), (250 / 255.0, 192 / 255.0, 15 / 255.0)])

# 绘制每个模型的 ROC 曲线
for model_idx, (fpr_model, tpr_model, roc_auc_model, style) in enumerate(model_results):
    # for i, color in zip(range(num_classes), colors):
    #     plt.plot(fpr_model[i], tpr_model[i], color=color, lw=lw,
    #              label=f'Model {model_idx + 1}, Class {i} (AUC = {roc_auc_model[i]:0.2f})')
    #
    # # 绘制微平均 ROC 曲线
    # plt.plot(fpr_model["micro"], tpr_model["micro"],
    #          label=f'Model {model_idx + 1}, micro-average (AUC = {roc_auc_model["micro"]:0.2f})',
    #          color='deeppink', linestyle=':', linewidth=4)

    # 绘制宏平均 ROC 曲线
    plt.plot(fpr_model["macro"], tpr_model["macro"],
             label=f'{models_name[model_idx].split("+", 1)[0]},  (AUC = {roc_auc_model["macro"]:0.2f})',
             color=style['color'], linestyle=style['linestyle'], linewidth=1)

# 随机猜测的对角线
#plt.plot([0, 1], [0, 1], '--', color='darkred', lw=2)

plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")

plt.rcParams['savefig.format'] = 'eps'
plt.savefig('./photo/roc_curve.tiff', format='tiff')
plt.savefig('./photo/roc_curve.eps', format='eps')
plt.show()

