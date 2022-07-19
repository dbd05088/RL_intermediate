import os
import numpy as np
import PIL
import torch

from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, selection,
                 root_dir: str,
                 transform=None):
        self.selection = selection
        self.root_dir = root_dir
        self.transform = transform
        self.n_per_cls = [len(cls_data) for cls_data in self.selection]
        self.num_samples = sum(self.n_per_cls)

        self.cls_name = os.listdir(root_dir)
        self.img_list = []
        self.label_list = []
        for i, cls in enumerate(self.cls_name):
            self.img_list += [os.path.join(root_dir, cls, f'{img_num+1:04}.png') for img_num in self.selection[i]]
            self.label_list += [i]*self.n_per_cls[i]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = PIL.Image.open(self.img_list[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_list[idx]
        sample = {'image': image, 'label': label}
        return sample


class TestDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cls_name = os.listdir(root_dir)
        self.img_list = []
        self.label_list = []
        for i, cls in enumerate(self.cls_name):
            cls_folder = os.path.join(root_dir, cls)
            cls_list = os.listdir(cls_folder)
            self.img_list += [os.path.join(cls_folder, cls_sample) for cls_sample in cls_list]
            self.label_list += [i]*len(cls_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = PIL.Image.open(self.img_list[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_list[idx]
        sample = {'filename': self.img_list[idx], 'image': image, 'label': label}
        return sample