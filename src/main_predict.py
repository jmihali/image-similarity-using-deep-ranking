import torch
from net import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from foodDataset import FoodDataset
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

net = TripletNet(resnet18())
net = torch.nn.DataParallel(net).to(device)
#weights = torch.load('/home/jmihali/Projects/checkpointcheckpoint.pth.tar', map_location=device)
weights = torch.load('/home/jmihali/Projects/image-similarity-using-deep-ranking/checkpoint/checkpoint_res18.pth.tar', map_location=device)
print('Number of epochs trained: ', weights['epoch'])
net.load_state_dict(weights['state_dict'])
net.eval()

data_transform = transforms.Compose([
    transforms.Resize((457, 308)),  # resizing
    transforms.ToTensor(),  # transform image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalization from [0, 1] to [-1, 1]
])

#### PREDICT ON TEST SET #####
print("Predicting on test set...")
batch_size=5
test_dataset = FoodDataset(file='/home/jmihali/Projects/image-similarity-using-deep-ranking/src/test_triplets.txt', root_dir='/home/jmihali/Projects/image-similarity-using-deep-ranking/food',
                           transform=data_transform, label=False)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

predictions = []
cnt=1
print('batch size =', batch_size)
for data in test_data_loader:
    print('Batch', cnt)
    cnt += 1
    img1_batch = data[0].to(device)
    img2_batch = data[1].to(device)
    img3_batch = data[2].to(device)

    features = net(img1_batch, img2_batch, img3_batch)

    for triple_features in zip(features[0], features[1], features[2]):
        if torch.dist(triple_features[0], triple_features[1]) < torch.dist(triple_features[0], triple_features[2]):
            predictions.append(1)
        else:
            predictions.append(0)

np.savetxt(fname='predictions_res18.txt', fmt="%d", X=predictions)
print("Prediction saved")
