"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import argparse
from utils import train, calculate_distance
from net import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from foodDataset import FoodDataset
import numpy as np


# Instantiate the parser
parser = argparse.ArgumentParser()

# directory
parser.add_argument('--ckptroot', type=str,
                    default="../checkpoint", help='path to checkpoint')
parser.add_argument('--dataroot', type=str, default="",
                    help='train/val data root')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float,
                    default=0.9, help='momentum factor')
parser.add_argument('--nesterov', type=bool, default=True,
                    help='enables Nesterov momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int,
                    default=30, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int,
                    default=30, help='test set input batch size')
parser.add_argument('--start_epoch', type=int,
                    default=0, help='starting epoch')

# loss function settings
parser.add_argument('--g', type=float, default=1.0, help='gap parameter')
parser.add_argument('--p', type=int, default=2,
                    help='norm degree for pairwise distance - Euclidean Distance')

# training settings
parser.add_argument('--resume', type=bool, default=False,
                    help='whether re-training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True,
                    help='whether training using GPU')

# parse the arguments
args = parser.parse_args()


def main():
    """Main pipeline of Image Similarity using Deep Ranking."""
    net = TripletNet(resnet101())

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu

    if args.is_gpu:
        print("==> Initialize CUDA support for TripletNet model ...")
        device = torch.device('cuda:0')
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = True


    """
    # resume training from the last time
    if args.resume:
        # Load checkpoint
        print('==> Resuming training from checkpoint ...')
        checkpoint = torch.load(args.ckptroot)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print("==> Loaded checkpoint '{}' (epoch {})".format(
            args.ckptroot, checkpoint['epoch']))

    else:
        # start over
        print('==> Building new TripletNet model ...')
        net = TripletNet(resnet101())
    """

    # Loss function, optimizer and scheduler
    criterion = nn.TripletMarginLoss(margin=args.g, p=args.p)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=10,
                                                           verbose=True)


    """
    # load triplet dataset
    trainloader, testloader = TinyImageNetLoader(args.dataroot,
                                                 args.batch_size_train,
                                                 args.batch_size_test)

    """

    # preparing dataloader

    # these are the transforms we make the food images
    data_transform = transforms.Compose([
        transforms.Resize((225, 225)),  # resizing
        transforms.ToTensor(),  # transform image to a tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalization from [0, 1] to [-1, 1]
    ])

    food_dataset = FoodDataset(file='/home/jmihali/Projects/image-similarity-using-deep-ranking/src/train_triplets.txt', root_dir='/home/jmihali/Projects/image-similarity-using-deep-ranking/food',
                               transform=data_transform, label=False)

    train_set_length = int(0.95 * len(food_dataset))
    val_set_length = len(food_dataset) - train_set_length
    print("Train set length:            ", train_set_length)
    print("Validation set length:       ", val_set_length)

    # in case you want to split the data:
    #train_data, val_data = random_split(food_dataset, [train_set_length, val_set_length])
    #trainloader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
    #testloader = DataLoader(val_data, batch_size=10, shuffle=True, num_workers=4)


    trainloader = DataLoader(food_dataset, batch_size=10, shuffle=True, num_workers=4)

    #train model
    train(net, criterion, optimizer, scheduler, trainloader, None, args.start_epoch, args.epochs, args.is_gpu)


    #### PREDICT ON TEST SET #####
    print("Predicting on test set...")
    test_dataset = FoodDataset(file='/home/jmihali/Projects/image-similarity-using-deep-ranking/src/test_triplets.txt', root_dir='/home/jmihali/Projects/image-similarity-using-deep-ranking/food',
                               transform=data_transform, label=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    net.eval()
    predictions = []
    for data in test_data_loader:
        img1_batch = data[0].to(device)
        img2_batch = data[1].to(device)
        img3_batch = data[2].to(device)

        f = net(img1_batch, img2_batch, img3_batch)
        if (torch.dist(f[0],f[1]) < torch.dist(f[0], f[2])):
            predictions.append(1)
        else:
            predictions.append(0)

    np.savetxt(fname='predictions.txt', fmt="%d", X=predictions)
    print("Prediction saved")


if __name__ == '__main__':
    main()
