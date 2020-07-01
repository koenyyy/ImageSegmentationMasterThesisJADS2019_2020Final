import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, k=1, channels_in=1, channels_out=2, kernel_size=3,
                 activation=F.leaky_relu, dropout=False, batchnorm=False, instancenorm=True,
                 padding=True, stride=1, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 pooling_kernel=2, pooling_stride=2):
        super(UNet, self).__init__()
        # initialize all variables
        self.k = k
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.instancenorm = instancenorm
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride

        self.pool1 = nn.MaxPool2d(self.pooling_kernel, self.pooling_stride)
        self.pool2 = nn.MaxPool2d(self.pooling_kernel, self.pooling_stride)
        self.pool3 = nn.MaxPool2d(self.pooling_kernel, self.pooling_stride)
        self.pool4 = nn.MaxPool2d(self.pooling_kernel, self.pooling_stride)

        self.UpConv1 = UpConv(16*self.k, 8*self.k, self.pooling_kernel, self.pooling_stride, self.padding)
        self.UpConv2 = UpConv(8*self.k, 4*self.k, self.pooling_kernel, self.pooling_stride, self.padding)
        self.UpConv3 = UpConv(4*self.k, 2*self.k, self.pooling_kernel, self.pooling_stride, self.padding)
        self.UpConv4 = UpConv(2*self.k, 1*self.k, self.pooling_kernel, self.pooling_stride, self.padding)

        self.UNetConvBlock1 = UNetConvBlock(self.channels_in, 1*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock2 = UNetConvBlock(self.k, 2*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock3 = UNetConvBlock(2*self.k, 4*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock4 = UNetConvBlock(4*self.k, 8*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock5 = UNetConvBlock(8*self.k, 16*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock6 = UNetConvBlock(16*self.k, 8*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock7 = UNetConvBlock(8*self.k, 4*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock8 = UNetConvBlock(4*self.k, 2*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)
        self.UNetConvBlock9 = UNetConvBlock(2*self.k, 1*self.k, self.kernel_size,
                                            self.activation, self.dropout, self.batchnorm, self.instancenorm, self.padding)

        self.finalConv = nn.Conv2d(1*self.k, self.channels_out, 1)

    def forward(self, x):
        res = self.UNetConvBlock1(x)
        skip1 = res.clone()
        res = self.pool1(res)

        res = self.UNetConvBlock2(res)
        skip2 = res.clone()
        res = self.pool2(res)

        res = self.UNetConvBlock3(res)
        skip3 = res.clone()
        res = self.pool3(res)

        res = self.UNetConvBlock4(res)
        skip4 = res.clone()
        res = self.pool4(res)

        res = self.UNetConvBlock5(res)
        res = self.UpConv1(res)
        skip4 = skip4
        res = torch.cat([res, skip4], dim=1)

        res = self.UNetConvBlock6(res)
        res = self.UpConv2(res)
        skip3 = skip3
        res = torch.cat([res, skip3], dim=1)

        res = self.UNetConvBlock7(res)
        res = self.UpConv3(res)
        skip2 = skip2
        res = torch.cat([res, skip2], dim=1)

        res = self.UNetConvBlock8(res)
        res = self.UpConv4(res)
        skip1 = skip1
        res = torch.cat([res, skip1], dim=1)

        res = self.UNetConvBlock9(res)

        res = self.finalConv(res)

        return res


class UNetConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, activation=F.leaky_relu, dropout=False,
                 batchnorm=False, instancenorm=True, padding=True):

        super(UNetConvBlock, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.instancenorm = instancenorm

        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm2d(channels_out)

        if padding:
            padding = 1
        else:
            padding = 0

        # Line below is not neccesary
        # self.conv = nn.Conv3d(channels_in, channels_out, kernel_size, padding=padding)

        if dropout:
            self.dropout_layer = nn.Dropout2d(p=0.2)

        if instancenorm:
            self.instance_layer = nn.InstanceNorm2d(channels_in)

        self.activation = activation
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size, padding=padding)

    def forward(self, x):
        # First convolution followed by activation function
        if self.dropout:
            x = self.dropout_layer(x)

        res = self.conv1(x)

        if self.batchnorm:
            res = self.batchnorm_layer(res)

        if self.instancenorm:
            res = self.instance_layer(res)

        res = self.activation(res)

        # Second convolution followed by activation function
        if self.dropout:
            res = self.dropout_layer(x)

        res = self.conv2(res)

        if self.batchnorm:
            res = self.batchnorm_layer(res)

        if self.instancenorm:
            res = self.instance_layer(res)

        res = self.activation(res)

        return res


class UpConv(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, stride, padding=True):
        super(UpConv, self).__init__()

        self.upsample = nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride)

    def forward(self, x):
        res = self.upsample(x)

        return res