#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

# Import network
from network import *
from imshow import *
from datasets import *

# Parser arguments
parser = argparse.ArgumentParser(description='Train PyTorch UNet with '
                                             'Massachusets Roads Dataset')
parser.add_argument('--train-percentage', '--t',
                    type=float, default=.2, metavar='N',
                    help='porcentage of the training set to use (default: .2)')
parser.add_argument('--batch-size', '--b',
                    type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--log-interval', '--li',
                    type=int, default=100, metavar='N',
                    help='how many batches to wait' +
                         'before logging training status')
parser.add_argument('--epochs', '--e',
                    type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--device', '--d',
                    default='cpu', choices=['cpu', 'cuda'],
                    help='pick device to run the training (defalut: "cpu")')
parser.add_argument('--network', '--n',
                    default='unet',
                    choices=['unet'],
                    help='pick a specific network to train (default: unet)')
parser.add_argument('--image-shape', '--imshape',
                    type=int, nargs='+',
                    default=[64, 64],
                    metavar='height width',
                    help='rectanlge size to crop input images '
                         '(default: 64 64)')
parser.add_argument('--filters', '--f',
                    type=int, default=16, metavar='N',
                    help='multiple of number of filters to use (default: 16)')
parser.add_argument('--optimizer', '--o',
                    default='sgd', choices=['adam', 'sgd'],
                    help='pick a specific optimizer (default: "adam")')
parser.add_argument('--dataset', '--data',
                    default='mass_roads',
                    choices=['mass_roads'],
                    help='pick a specific dataset (default: "mass_roads")')
parser.add_argument('--normalize', '--norm',
                    action='store_true',
                    help='normalize images and use tanh')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
args = parser.parse_args()
print(args)


def train(trainset):

    # Split datasetO
    train_size = int(args.train_percentage * len(trainset))
    test_size = len(trainset) - train_size
    train_dataset, test_dataset \
        = torch.utils.data.random_split(trainset, [train_size, test_size])

    # Dataset information
    print('train dataset : {} elements'.format(len(train_dataset)))

    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True)
    args.dataset_size = len(train_loader.dataset)
    args.dataloader_size = len(train_loader)

    # get some random training images
    dataiter = iter(train_loader)
    inpt, trgt = dataiter.next()

    # Image range from (-1,1) to (0,1)
    grid = torchvision.utils.make_grid(
        torch.cat((inpt, trgt)),
        nrow=args.batch_size)

    # If images were normalized
    if args.normalize:
        grid = 0.5 * (grid + 1.0)

    # Write sample to tensorboard
    args.writer.add_image('sample-train', grid)

    # Show sample of images
    if args.plot:
        imshow(grid)

    # Define optimizer
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam(args.net.parameters(), lr=.0001)
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD(args.net.parameters(),
                                   lr=0.01, momentum=0.9)

    # Set loss function
    criterion = torch.nn.MSELoss()

    # Restore past checkpoint
    restore_checkpoint()

    # Set best for minimization
    args.best = float('inf')

    print('Started Training')
    # loop over the dataset multiple times
    for epoch in range(args.epochs):

        # reset running loss statistics
        args.train_loss = args.running_loss = 0.0

        for batch_idx, data in enumerate(train_loader, 1):

            # Get disparity data
            inputs, targets = data

            # Squeeze dimension
            targets = targets.mean(dim=1, keepdim=True)

            # Send to device
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # Calculate gradients and update
            with autograd.detect_anomaly():
                # zero the parameter gradients
                args.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = args.net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                args.optimizer.step()

            # Global step
            global_step = batch_idx + len(train_loader) * epoch

            # update running loss statistics
            args.running_loss += loss.item()
            args.train_loss += loss.item()

            # Write tensorboard statistics
            args.writer.add_scalar('Train/loss', loss.item(), global_step)

            # print every args.log_interval of batc|hes
            if batch_idx % args.log_interval == 0:
                print('Train Epoch : {} Batches : {} [{}/{} ({:.0f}%)]'
                      '\tLoss : {:.8f}'
                      .format(epoch, batch_idx,
                              args.batch_size * batch_idx,
                              args.dataset_size,
                              100. * batch_idx / args.dataloader_size,
                              args.running_loss / args.log_interval))

                args.running_loss = 0.0

                # Add images to tensorboard
                write_images_to_tensorboard(targets,
                                            outputs,
                                            global_step,
                                            step=True)

                # Process current checkpoint
                process_checkpoint(loss.item(), targets, outputs, global_step)

        print('====> Epoch: {} Average loss: {:.4f}'
              .format(epoch, args.train_loss / len(train_loader)))

    # Add trained model
    print('Finished Training')


def restore_checkpoint():
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore past checkpoint
    args.net.load_state_dict(checkpoint['net_state_dict'])
    args.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # To continue training
    args.net.train()


def process_checkpoint(loss, targets, outputs, global_step):

    # check if current batch had best generating fitness
    steps_before_best = 100
    if loss < args.best and global_step > steps_before_best:
        args.best = loss

        # Save best checkpoint
        torch.save({
            'net_state_dict': args.net.state_dict(),
            'optimizer_state_dict': args.optimizer.state_dict(),
        }, "checkpoint/best_{}.pt".format(args.run))

        # Write tensorboard statistics
        args.writer.add_scalar('Best/loss', loss, global_step)

        # Save best generation image
        write_images_to_tensorboard(targets, outputs, global_step,
                                    best=True)

    # Save current checkpoint
    torch.save({
        'state_dict': args.net.state_dict(),
        'optimizer_state_dict': args.optimizer.state_dict(),
    }, "checkpoint/last_{}.pt".format(args.run))


def write_images_to_tensorboard(targets, outputs, global_step,
                                step=False, best=False):
    # Add images to tensorboard
    # Current network fit
    grid = torchvision.utils.make_grid(torch.cat((
        outputs.cpu(), targets.cpu())), nrow=args.batch_size)

    # If images were normalized
    if args.normalize:
        grid = 0.5 * (grid + 1.0)

    if step:
        args.writer.add_image('Train/fit', grid, global_step)
    elif best:
        args.writer.add_image('Best/fit', grid, global_step)
    else:
        args.writer.add_image('fit', grid)
        imshow(grid)


def create_run_name():
    run = '{}={}'.format('nw', args.network)
    run += '_{}={}'.format('ds', args.dataset)
    run += '_{}={}'.format('fl', args.filters)
    run += '_{}={}'.format('op', args.optimizer)
    run += '_{}={}'.format('ep', args.epochs)
    run += '_{}={}'.format('bs', args.batch_size)
    run += '_{}={}'.format('tp', args.train_percentage)
    run += '_{}={}'.format('nm', 't' if args.normalize else 'f')
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    return run


def main():
    # Save parameters in string to name the execution
    args.run = create_run_name()

    # print run name
    print('execution name : {}'.format(args.run))

    # Tensorboard summary writer
    args.writer = SummaryWriter('runs/' + args.run)

    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=5)

    # Set up GPU
    if args.device != 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # Dataset information
    print('device : {}'.format(args.device))

    # Set dataset transform
    transform = transforms.Compose([
        transforms.Resize(args.image_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        if args.normalize
        else nn.Identity()
    ])

    # Load dataset
    if args.dataset == 'mass_roads':
        trainset = Mass_Roads(transform, train=True)

    # If images were normalized
    if args.normalize:
        args.activation = 'tanh'
    else:
        args.activation = 'sigmoid'

    # Create network
    if args.network == 'unet':
        # Add 3 color channels (left + right)
        net = UNet([3] + args.image_shape,
                   args.filters, args.activation)

    # Send networks to device
    args.net = net.to(args.device)

    if args.summary:
        print(args.net)

    # Train network
    train(trainset)

    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()
