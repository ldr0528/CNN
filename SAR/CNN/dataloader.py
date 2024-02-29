import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = sorted(os.listdir(os.path.join(root_dir, split)))
        self.data = self.load_data()

    def load_data(self):
        data = []
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, self.split, class_dir)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                np_data = np.load(file_path)
                data.append((np_data, self.classes.index(class_dir)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        np_data, label = self.data[index]
        np_data = self.transform(np_data).float()
        # 如果需要进行数据增强或其他预处理，可以在这里添加相应的代码
        return np_data, label



# class CustomDataset(Dataset):
#     def __init__(self, root_dir, split='train', transform=None):
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform
#         self.classes = sorted(os.listdir(os.path.join(root_dir, split)))
#
#     def __len__(self):
#         total_samples = 0
#         for class_dir in self.classes:
#             class_path = os.path.join(self.root_dir, self.split, class_dir)
#             total_samples += len(os.listdir(class_path))
#         return total_samples
#
#     def __getitem__(self, index):
#         class_idx = 0
#         while index >= len(os.listdir(os.path.join(self.root_dir, self.split, self.classes[class_idx]))):
#             index -= len(os.listdir(os.path.join(self.root_dir, self.split, self.classes[class_idx])))
#             class_idx += 1
#
#         class_dir = self.classes[class_idx]
#         class_path = os.path.join(self.root_dir, self.split, class_dir)
#         file_name = os.listdir(class_path)[index]
#         file_path = os.path.join(class_path, file_name)
#
#         data = np.load(file_path)
#
#         if self.transform:
#             data = self.transform(data).float()
#
#
#         label = class_idx
#
#         return data, label