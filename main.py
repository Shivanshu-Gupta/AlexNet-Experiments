import os
import argparse
import pprint
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
from dataset import AlexnetDataset, imshow, remove_exif
from network import alexnet
from train import train_model

pp = pprint.PrettyPrinter()

plt.ion()

parser = argparse.ArgumentParser(description='Alexnet Experiment')
parser.add_argument('--data_root', default='/home/cse/dual/cs5130298/scratch/ImageNet_Subset/')
parser.add_argument('--remove_exif', action='store_true', default=False)
parser.add_argument('--use_gpu', action='store_true', default=False)
parser.add_argument('--epochs', default=40, type=int, metavar='N')
parser.add_argument('--lr', default=0.01, type=float, metavar='LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--step_size', default=10, type=int, metavar='N')
parser.add_argument('--save_dir', default='checkpoints/', type=str, metavar='PATH')
parser.add_argument('--reload', default='', type=str, metavar='PATH',
                    help='path to checkpoint to load (default: none)')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on test set')
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


def save_params():
    os.makedirs(args.save_dir, exist_ok=True)
    param_file = args.save_dir + '/' + 'params.pt'
    torch.save(args, param_file)


if __name__ == '__main__':
    pp.pprint(args)
    save_params()

    if args.remove_exif:
        for phase in phases:
            remove_exif(root, phase)
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    if args.reload:
        if os.path.isfile(args.reload):
            print("=> loading checkpoint '{}'".format(args.reload))
            checkpoint = torch.load(args.reload)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.reload_state_dict(checkpoint['state_dict'])
            # optimizer.reload_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.reload, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.reload))
    else:
        model = alexnet(pretrained=False, num_classes=len(class_names))

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=1)

    # TODO: Store the parameters and use them to initialise next time.
    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, args.save_dir,
        num_epochs=args.epochs, use_gpu=use_gpu)
