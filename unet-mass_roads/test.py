#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import math
from datetime import datetime

import torch
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms

# Import network
from network import *
from imshow import *
from datasets import *


# Parser arguments
parser = argparse.ArgumentParser(description='Test PyTorch UNet with '
                                             'Tsukuba Disparity Maps')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored for inference')
parser.add_argument('--batch-size', '--b',
                    type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16)')
parser.add_argument('--device', '--d',
                    default='cuda', choices=['cpu', 'cuda'],
                    help='pick device to run the training (defalut: "cpu")')
parser.add_argument('--network', '--n',
                    default='unet',
                    choices=['unet', 'dunet'],
                    help='pick a specific network to train (default: dunet)')
parser.add_argument('--image-shape', '--imshape',
                    type=int, nargs='+',
                    default=[64, 64],
                    metavar='height width',
                    help='rectanlge size to crop input images '
                         '(default: 64 64)')
parser.add_argument('--filters', '--f',
                    type=int, default=16, metavar='N',
                    help='multiple of number of filters to use (default: 16)')
parser.add_argument('--dataset', '--data',
                    default='tsukuba',
                    choices=['tsukuba'],
                    help='pick a specific dataset (default: "tsukuba")')
parser.add_argument('--normalize', '--norm',
                    action='store_true',
                    help='normalize images and use tanh')
parser.add_argument('--sample-size', '--s',
                    type=int, default=16, metavar='N',
                    help='sample size for generating images (default: 16)')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
args = parser.parse_args()
print(args)


def test(testset):
    # Create dataset loader
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False)

    args.dataset_size = len(test_loader.dataset)
    args.dataloader_size = len(test_loader)

    # get some random training images
    dataiter = iter(test_loader)

    # Get fifth batch for testing and comparing
    for i in range(5):
        left, right, disp = dataiter.next()

    disp = torch.cat((disp,) * 3, dim=1)

    # Image range from (-1,1) to (0,1)
    grid = torchvision.utils.make_grid(
        torch.cat((left, right, disp)),
        nrow=args.batch_size)

    # If images were normalized
    if args.normalize:
        grid = 0.5 * (grid + 1.0)

    # Show sample of images
    if args.plot:
        imshow(grid)

    # Restore past checkpoint
    restore_checkpoint()

    # Concatenate left and right
    inpt = torch.cat((left, right), dim=1)

    # Send to device
    inpt = inpt.to(args.device)
    disp = disp.to(args.device)

    # Forward through network
    with torch.no_grad():
        outpt = args.net(inpt)

    outpt = torch.cat((outpt,) * 3, dim=1)

    # Create grid
    grid = torchvision.utils.make_grid(
        torch.cat((left.cpu(), right.cpu(), disp.cpu(), outpt.cpu())),
        nrow=4)

    # If images were normalized
    if args.normalize:
        grid = 0.5 * (grid + 1.0)

    # Show sample of images
    if args.plot:
        imshow(grid)

    # Set acurracy function
    criterion = torch.nn.MSELoss()

    # Test accuracy
    args.test_acc = 0.0

    # Test network accuracy
    for batch_idx, data in enumerate(test_loader, 1):

        # Get disparity data
        tLeft, tRight, targets = data

        # Concatenate left and right
        inputs = torch.cat((tLeft, tRight), dim=1)

        # Send to device
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # Forward through network
        with torch.no_grad():
            outputs = args.net(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Sum global test accuracy
        args.test_acc += loss.item()

        print('Batch : {} [{}/{} ({:.0f}%)]'
              '\tLoss : {:.8f}'
              .format(batch_idx,
                      args.batch_size * batch_idx,
                      args.dataset_size,
                      100. * batch_idx / args.dataloader_size,
                      loss.item()))

    print('====> Average Accuracy: {:.4f}'.format(
        args.test_acc / len(test_loader)))


def restore_checkpoint():
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore past checkpoint
    args.net.load_state_dict(checkpoint['state_dict'])

    # To do inference
    args.net.eval()


def main():
    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=5)

    # Set up GPU
    if args.device != 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
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
    if args.dataset == 'tsukuba':
        testset = Tsukuba(transform, train=False)

    # If images were normalized
    if args.normalize:
        args.activation = 'tanh'
    else:
        args.activation = 'sigmoid'

    # Create network
    if args.network == 'unet':
        # Add 6 color channels (left + right)
        net = UNet([6] + args.image_shape,
                   args.filters, args.activation)
    elif args.network == 'dunet':
        # Add 6 color channels (left + right)
        net = UNet_Disparity([6] + args.image_shape,
                             args.filters, args.activation)

    # Send networks to device
    args.net = net.to(args.device)

    if args.summary:
        print("Network")
        summary(args.net, input_size=tuple([6] + args.image_shape))
        print()

    # Test the trained model if provided
    if args.checkpoint != 'none':
        test(testset)


if __name__ == "__main__":
    main()
