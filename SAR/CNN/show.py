import itertools
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from numpy import interp
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from dataloader import CustomDataset
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.metrics import recall_score, precision_score, roc_auc_score

# from models.coatnet import coatnet_0
from models.resnet18 import resnet18_1
from models.resnet18_3 import resnet18_3
from models.densenet121 import densenet121
from models.densenet121_3 import densenet121
from models.mobilenetv3_3 import mobilenetv3
from models.efficientnet_3 import efficientnetb0
from models.vgg16_3 import vgg16

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
])


batch_size = 8
num_classes = 3

num_workers = 0

# 定义测试数据集
#root_dir = 'fusarship'

# 定义测试数据集的dataloader
#est_dataset = CustomDataset(root_dir, split='Test', transform=transform)
#test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

root_dir = 'OSSHIP1/Test'
test_dataset = ImageFolder(root=root_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#model = resnet18_3(num_classes=num_classes)
#model = vgg16(num_classes=num_classes)
#model = efficientnetb0(num_classes=num_classes)
#model = mobilenetv3(num_classes=num_classes)
model = densenet121(num_classes=num_classes)
# model = coatnet_0(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 0
#checkpoint = torch.load('checkpoint/densenet+resnet-18+hog.pkl', map_location=torch.device('cpu'))
checkpoint = torch.load('checkpoint3/densenet-121.pkl')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.cuda()


# 设置模型为评估模式
model.eval()

# 初始化测试损失为0.0
test_loss = 0.0
correct = 0
total = 0

# 初始化混淆矩阵
all_labels = []
all_predicted_probs = []
conf_matrix = np.zeros((num_classes, num_classes))

with torch.no_grad():  # 不计算梯度值，节省内存空间和计算时间
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
        conf_matrix += confusion_matrix(labels.cpu(), predicted.cpu(), labels=np.arange(num_classes))

        # 收集预测概率用于计算ROC曲线
        probs = torch.nn.functional.softmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_predicted_probs.extend(probs.cpu().numpy())

test_loss /= len(test_dataset)  # 计算平均测试损失值
accuracy = correct / total
print(f"测试损失为{test_loss:.4f}, 准确率为{accuracy}")

classes = ["Fishing", "Passenger", "Tanker"]

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(num_classes), classes, rotation=45)
plt.yticks(np.arange(num_classes), classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
# 添加数量标签
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center',
                 color='black')
plt.show()

# 添加绘制ROC曲线和计算F1-score的代码
prediction = np.array(all_predicted_probs)
labels = torch.tensor(all_labels)
labels = labels.reshape((labels.shape[0], 1))
labels_onehot = torch.zeros(labels.shape[0], num_classes)
labels_onehot.scatter_(dim=1, index=labels, value=1)
labels_onehot = np.array(labels_onehot)


# 添加绘制ROC曲线和计算F1-score的代码
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(labels_onehot.ravel(), prediction.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()

lw = itertools.cycle([2.9, 2.2, 2.2, 0.8])
colors = itertools.cycle([(69 / 255.0, 189 / 255.0, 155 / 255.0), (43 / 255.0, 85 / 255.0, 125 / 255.0),
                          (240 / 255.0, 81 / 255.0, 121 / 255.0), (250 / 255.0, 192 / 255.0, 15 / 255.0)])
# ----------------------------------------
# 绘制所有类的曲线
# ----------------------------------------
for i, color, l in zip(range(num_classes), colors, lw):
    plt.plot(fpr[i], tpr[i], color=color, lw=l,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], '--', lw=2, color='darkred')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()


# 将预测结果和真实标签转换为 NumPy 数组
prediction = np.array(all_predicted_probs)
labels = np.array(all_labels)

# 计算 F1-Score
f1_scores = f1_score(labels, np.argmax(prediction, axis=1), average=None)
recall_scores = recall_score(labels, np.argmax(prediction, axis=1), average=None)
precision_scores = precision_score(labels, np.argmax(prediction, axis=1), average=None)

# 输出每个类别的 F1-Score
for i, f1 in enumerate(f1_scores):
    print("Class {} - F1-Score: {:.2f}".format(i, f1))

# 输出每个类别的 F1-Score, Recall, Precision
for i in range(num_classes):
    print(f"Class {i}:")
    print(f"  F1-Score: {f1_scores[i]:.2f}")
    print(f"  Recall: {recall_scores[i]:.2f}")
    print(f"  Precision: {precision_scores[i]:.2f}")

# 计算平均 F1-Score F1-Score, Recall, Precision
average_f1 = f1_score(labels, np.argmax(prediction, axis=1), average='macro')
average_recall = np.mean(recall_scores)
average_precision = np.mean(precision_scores)

print(f"Average F1-Score: {average_f1:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average Precision: {average_precision:.4f}")

sensitivity = np.zeros(num_classes)  # 初始化敏感性数组
specificity = np.zeros(num_classes)  # 初始化特异性数组
LR_plus = np.zeros(num_classes)      # 初始化似然比阳性数组

for i in range(num_classes):
    TP = conf_matrix[i, i]  # 真阳性：混淆矩阵中对角线上的值
    FN = np.sum(conf_matrix[i, :]) - TP  # 假阴性：该行的总和减去真阳性
    TN = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + TP  # 真阴性
    FP = np.sum(conf_matrix[:, i]) - TP  # 假阳性：该列的总和减去真阳性

    sensitivity[i] = TP / (TP + FN)  # 计算敏感性（真阳性率）
    specificity[i] = TN / (TN + FP)  # 计算特异性（真阴性率）
    LR_plus[i] = sensitivity[i] / (1 - specificity[i])  # 计算似然比阳性

average_LR_plus = np.mean(LR_plus)  # 计算平均似然比阳性

print(f"Average LR+: {average_LR_plus:.2f}")
for i in range(num_classes):
    print(f"Class {i} - Sensitivity: {sensitivity[i]:.2f}, Specificity: {specificity[i]:.2f}, LR+: {LR_plus[i]:.2f}")

