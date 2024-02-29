import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


# RandAugment 数据增强
class RandAugment:
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m

    def __call__(self, img):
        ops = self._get_transforms()
        for op, magnitude in ops:
            img = op(img, magnitude)
        return img

    def _get_transforms(self):
        transforms_list = [
            (transforms.RandomHorizontalFlip, 0),
            (transforms.RandomVerticalFlip, 0),
            (transforms.RandomAffine, self.m),
            (transforms.ColorJitter, self.m),
            (transforms.RandomRotation, self.m)
        ]

        ops = np.random.choice(len(transforms_list), self.n, replace=False)
        return [(transforms_list[op][0](transforms_list[op][1]), np.random.randint(0, self.m)) for op in ops]


# MixUp 数据增强
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MixUpDataset(Dataset):
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.alpha = alpha

    def __getitem__(self, index):
        x, y = self.dataset[index]
        mixed_x, y_a, y_b, lam = mixup_data(x, y, self.alpha)
        return mixed_x, y_a, y_b, lam

    def __len__(self):
        return len(self.dataset)
