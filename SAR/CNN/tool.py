import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# 加载原始数据集
# train_dataset = ImageFolder(root='../dataset/OpenSARship_update/train', transform=None)

# 定义数据增强的转换列表
augmentation_transforms = [
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=210),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomRotation(degrees=120),
    # 添加其他数据增强方法
]

# 创建保存数据的文件夹
save_path = '../dataset/OpenSARship_update/train'
if not os.path.exists(save_path):
    os.makedirs(save_path)
class_name = os.listdir(save_path)
# 逐个应用数据增强方法并保存

# 检查每个类别的数据是否已经达到3000条

for transform_idx, augmentation_transform in enumerate(augmentation_transforms):
    augmented_dataset = ImageFolder(root=save_path, transform=augmentation_transform)

    # 使用identity collate_fn确保不改变图像的类型
    dataloader = DataLoader(augmented_dataset, batch_size=1, collate_fn=lambda x: x[0])

    for idx, (image, label) in enumerate(dataloader):
        # 获取对应的类别文件夹
        class_folder = os.path.join(save_path, f"{class_name[label]}")
        if len(os.listdir(class_folder)) >= 14000:
            continue
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # 使用transforms.Compose将PIL.Image.Image转换为张量
        to_tensor = transforms.Compose([
            transforms.ToTensor()])
        image = to_tensor(image)  # 由于batch_size=1，需要去除批次维度

        # 保存图像到对应的类别文件夹
        save_image(image, os.path.join(class_folder, f"augmented_{transform_idx}_{idx}.jpg"))

    print(f"Augmentation {transform_idx + 1} results saved to:", save_path)
