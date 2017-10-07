import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from dataset import AlexnetDataset, imshow
from network import alexnet
from train import train_model

plt.ion()

parser = argparse.ArgumentParser(description='Alexnet Experiment')
parser.add_argument('--data_root', dest='data_root', default='/home/cse/dual/cs5130298/scratch/ImageNet_Subset/')
parser.add_argument('--remove_exif', dest='remove_exif', action='store_true', default=False)
parser.add_argument('--usegpu', dest='use_gpu', action='store_true', default=False)
parser.add_argument('--epochs', dest='num_epochs', type=int, default=20)
parser.add_argument('--lr', dest='lr', type=float, default=0.01)
parser.add_argument('--step', dest='step_size', type=int, default=10)
args = parser.parse_args()

use_gpu = args.use_gpu

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
root = args.data_root
phases = ['train', 'test', 'validation']
image_datasets = {x: AlexnetDataset(root, x, data_transforms[x]) for x in phases}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in phases}
dataset_sizes = {x: len(image_datasets[x]) for x in phases}
class_names = image_datasets['train'].classes

print(class_names)
print(dataset_sizes)

if __name__=='__main__':
    if args.remove_exif:
        for phase in phases:
            remove_exif(root, phase)
    model = alexnet(pretrained=False, num_classes=len(class_names))

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Decay LR by a factor of gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=1)
    # TODO: Store the parameters and use them to initialise next time.
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler,
                           num_epochs=args.num_epochs, use_gpu=use_gpu)
