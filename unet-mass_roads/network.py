import math

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, image_shape, filters=8, activation='sigmoid'):
        super(UNet, self).__init__()
        self.c, self.h, self.w = image_shape
        f = self.f = filters
        self.activation = activation

        # Level 1
        self.left_conv_block1 = self.conv_block(self.c, f)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block1 = self.conv_block(f * 2, f)
        self.conv_output = nn.Conv2d(f, 1, kernel_size=1, padding=0)

        # Level 2
        self.left_conv_block2 = self.conv_block(f, f * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block2 = self.conv_block(f * 4, f * 2)
        self.tconv2 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)

        # Level 3
        self.left_conv_block3 = self.conv_block(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block3 = self.conv_block(f * 8, f * 4)
        self.tconv3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)

        # Level 4
        self.left_conv_block4 = self.conv_block(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block4 = self.conv_block(f * 16, f * 8)
        self.tconv4 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)

        # Level 5 (BottleNeck)
        self.left_conv_block5 = self.conv_block(f * 8, f * 16)
        self.tconv5 = nn.ConvTranspose2d(f * 16, f * 8,
                                         kernel_size=2,
                                         stride=2)

        # Intialize weights
        self.apply(self.initialize_weights)

    def conv_block(self, in_chan, out_chan, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU()
        )

    def initialize_weights(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            input_dimension = module.in_channels \
                * module.kernel_size[0] \
                * module.kernel_size[1]
            std_dev = math.sqrt(2.0 / float(input_dimension))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)

    def forward(self, x):

        # Level 1
        x1 = self.left_conv_block1(x)
        # Downsample
        x2 = self.pool1(x1)

        # Level 2
        x2 = self.left_conv_block2(x2)
        # Downsample
        x3 = self.pool2(x2)

        # Level 3
        x3 = self.left_conv_block3(x3)
        # Downsample
        x4 = self.pool3(x3)

        # Level 4
        x4 = self.left_conv_block4(x4)
        # Downsample
        x5 = self.pool4(x4)

        # Level 5
        x5 = self.left_conv_block5(x5)
        # Upsample
        x6 = self.tconv5(x5)

        # Level 4
        x6 = torch.cat((x6, x4), 1)
        x6 = self.right_conv_block4(x6)
        # Upsample
        x7 = self.tconv4(x6)

        # Level 3
        x7 = torch.cat((x7, x3), 1)
        x7 = self.right_conv_block3(x7)
        # Upsample
        x8 = self.tconv3(x7)

        # Level 2
        x8 = torch.cat((x8, x2), 1)
        x8 = self.right_conv_block2(x8)
        # Upsample
        x9 = self.tconv2(x8)

        # Level 1
        x9 = torch.cat((x9, x1), 1)
        x9 = self.right_conv_block1(x9)

        if self.activation == 'sigmoid':
            x_out = torch.sigmoid(self.conv_output(x9))
        elif self.activation == 'tanh':
            x_out = torch.tanh(self.conv_output(x9))

        return x_out
