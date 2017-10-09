import os
import argparse
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import AlexnetDataset, remove_exif
from network import alexnet
from train import train_model, test_model

pp = pprint.PrettyPrinter()

plt.ion()

parser = argparse.ArgumentParser(description='Alexnet Experiment')
parser.add_argument('--data_root', default='/home/cse/dual/cs5130298/scratch/ImageNet_Subset/')
parser.add_argument('--remove_exif', action='store_true', default=False)
parser.add_argument('--use_gpu', action='store_true', default=False)

# experiment options
parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu',
                    help='activation function to use. (default: relu)')
parser.add_argument('--no_dropout', action='store_true', default=False)
parser.add_argument('--no_overlap', action='store_true', default=False)
parser.add_argument('--optimizer', type=str, choices=['sgdmomwd', 'sgd', 'sgdmom', 'adam'], default='sgdmomwd',
                    help='optimizer to use. (default: sgdmomwd)')

parser.add_argument('--epochs', type=int, default=40, metavar='N')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--step_size', type=int, default=10, metavar='N')
parser.add_argument('--init_wts', action='store_true', default=False)
# saving and reloading
parser.add_argument('--save_dir', type=str, default='checkpoints/', metavar='PATH')
parser.add_argument('--reload', type=str, default='', metavar='PATH',
                    help='path to checkpoint to load (default: none)')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='Manual epoch number (useful on restarts)')

# to run in test mode
parser.add_argument('--test', default=False, action='store_true',
                    help='test model on test set (use with --reload)')
args = parser.parse_args()

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


def save_params():
    os.makedirs(args.save_dir, exist_ok=True)
    param_file = args.save_dir + '/' + 'params.pt'
    torch.save(args, param_file)


def load_datasets(phases):
    image_datasets = {x: AlexnetDataset(args.data_root, x, data_transforms[x]) for x in phases}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in phases}
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    class_names = image_datasets[phases[0]].classes
    print(class_names)
    print(dataset_sizes)
    return dataloaders, class_names


if __name__ == '__main__':
    pp.pprint(args)
    save_params()

    if not args.test:
        phases = ['train', 'test', 'validation']
    else:
        phases = ['test']

    if args.remove_exif:
        for phase in phases:
            remove_exif(args.data_root, phase)

    dataloaders, class_names = load_datasets(phases)
    dropout = (not args.no_dropout)
    overlap = (not args.overlap)
    model = alexnet(pretrained=False, num_classes=len(class_names),
                    relu=(args.activation == 'relu'), dropout=dropout,
                    overlap=overlap, init_wts=args.init_wts)
    if args.reload:
        if os.path.isfile(args.reload):
            print("=> loading checkpoint '{}'".format(args.reload))
            checkpoint = torch.load(args.reload)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.reload_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, accuracy {})"
                  .format(args.reload, checkpoint['epoch'], checkpoint['best_acc']))
        else:
            print("=> no checkpoint found at '{}'".format(args.reload))

    if args.use_gpu:
        model = model.cuda()

    if args.test:
        test_model(model, dataloaders['test'], use_gpu=args.use_gpu)
    else:
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgdmom':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

        # Decay LR by a factor of gamma every step_size epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=1)

        # TODO: Store the parameters and use them to initialise next time.
        model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, args.save_dir,
                            num_epochs=args.epochs, use_gpu=args.use_gpu)
