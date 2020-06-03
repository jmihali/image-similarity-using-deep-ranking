import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os

class FoodDataset(Dataset):
    def __init__(self, file, root_dir, transform=None, label=True):
        self.food_frame = pd.read_csv(file, dtype=str, sep=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.food_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1_name = os.path.join(self.root_dir, self.food_frame.iloc[idx, 0]) + '.jpg' #str(self.food_frame.iloc[idx, 0]) + '.jpg'
        img2_name = os.path.join(self.root_dir, self.food_frame.iloc[idx, 1]) + '.jpg'
        img3_name = os.path.join(self.root_dir, self.food_frame.iloc[idx, 2]) + '.jpg'

        if self.label:
            label = int(self.food_frame.iloc[idx, 3])

        img1 = Image.open(img1_name)
        img2 = Image.open(img2_name)
        img3 = Image.open(img3_name)

        if self.label:
            sample = {'img1': img1, 'img2': img2, 'img3': img3, 'label': label}
        else:
            sample = {'img1': img1, 'img2': img2, 'img3': img3}

        if self.transform:
            sample['img1'] = self.transform(sample['img1'])
            sample['img2'] = self.transform(sample['img2'])
            sample['img3'] = self.transform(sample['img3'])

        return sample['img1'], sample['img2'], sample['img3']


