import torch
from net import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from foodDataset import FoodDataset
import numpy as np

predictions = []

predictions.append(1)
predictions.append(2)
predictions.append(7)
predictions.append(4)
# for data in test_data_loader:
#     img1_batch = data[0].to(device)
#     img2_batch = data[1].to(device)
#     img3_batch = data[2].to(device)
#
#     features = net(img1_batch, img2_batch, img3_batch)
#
#     for triple_features in zip(features[0], features[1], features[2]):
#         if torch.dist(triple_features[0], triple_features[1]) < torch.dist(triple_features[0], triple_features[2]):
#             predictions.append(1)
#         else:
#             predictions.append(0)

np.savetxt(fname='test.txt', fmt="%d", X=predictions)
print("Prediction saved")