import shutil
import time
import torch
from torch.autograd import Variable

def train(model, train_loader, criterion, optimizer, use_gpu=False):
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for (inputs, labels) in train_loader:
        # wrap input, labels in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / len(train_loader)
    acc = running_corrects / len(train_loader)

    print('Train Loss: {:.4f} Acc: {:.4f} ({}/{})'.format(loss, acc, running_corrects, len(train_loader)))
    return loss, acc


def validate(model, val_loader, criterion, use_gpu=False):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for (inputs, labels) in val_loader:
        # wrap input, labels in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
    loss = running_loss / len(val_loader)
    acc = running_corrects / len(val_loader)
    print('Validation Loss: {:.4f} Acc: {:.4f} ({}/{})'.format(loss, acc, running_corrects, len(val_loader)))
    return loss, acc


def train_model(model, data_loaders, criterion, optimizer, scheduler, save_dir, num_epochs=25, use_gpu=False):
    print('Training Model with use_gpu={}...'.format(use_gpu))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        train(model, data_loaders['train'], criterion, optimizer, use_gpu)
        epoch_loss, epoch_acc = validate(model, data_loaders['validation'], criterion, use_gpu)
        # TODO: Use TensorBoard to record metrics.
        # deep copy the model
        is_best = epoch_acc > best_acc
        if is_best:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        save_checkpoint(save_dir, {
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
        }, is_best)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(save_dir, state, is_best):
    savepath = save_dir + '/' + 'checkpoint.pth.tar'
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, save_dir + '/' + 'model_best.pth.tar')


def test_model(model, test_loader, use_gpu=False):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    # Iterate over data.
    for (inputs, labels) in test_loader:
        # wrap input, labels in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # statistics
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects / len(test_loader)
    print('Test Acc: {:.4f} ({}/{})'.format(acc, running_corrects, len(test_loader)))
    return acc
