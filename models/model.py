import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import src.data.dataloader as dl
import src.utils as utils


#    ______                    _    _
#   |  ____|                  | |  (_)
#   | |__  _   _  _ __    ___ | |_  _   ___   _ __   ___
#   |  __|| | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
#   | |   | |_| || | | || (__ | |_ | || (_) || | | |\__ \
#   |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
#
#

#    ___         _            __                  _    _
#   |   \  __ _ | |_  __ _   / _| _  _  _ _   __ | |_ (_) ___  _ _   ___
#   | |) |/ _` ||  _|/ _` | |  _|| || || ' \ / _||  _|| |/ _ \| ' \ (_-<
#   |___/ \__,_| \__|\__,_| |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
#

def adjust_sleep_data_dim(data, to_image_dim=20, is_psd=False):
    """
    Reshape: tensor.Size(batch_size, ch_count, sleep_epoch=3000) to tensor.Size(batch_size, ch_count, 20, 150)
    Args:
        data: x data from Dataloader
        to_image_dim: dimension in which we want to reshape 1D data to image like dimensions

    Returns:
        x from dataloader.__getitem__ method reshaped to fit MNIST model conv layers. Kernel can't be larger
        than x dimensions
    """
    if is_psd:
        data = torch.transpose(data, 2, 1).unsqueeze(3)
        data = data.float()
        return data

    if data.shape[1] > 1:
        data = data.unsqueeze(2)
    else:
        data = data.unsqueeze(1)
    # reshaping to fit conv layers kernel size
    last_dime_size = data.shape[2] * data.shape[3]

    data = torch.reshape(data, (data.shape[0], data.shape[1], to_image_dim, int(last_dime_size / to_image_dim)))

    return data


#    __  __          _       _    __                  _    _
#   |  \/  | ___  __| | ___ | |  / _| _  _  _ _   __ | |_ (_) ___  _ _   ___
#   | |\/| |/ _ \/ _` |/ -_)| | |  _|| || || ' \ / _||  _|| |/ _ \| ' \ (_-<
#   |_|  |_|\___/\__,_|\___||_| |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
#
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#     _____  _
#    / ____|| |
#   | |     | |  __ _  ___  ___   ___  ___
#   | |     | | / _` |/ __|/ __| / _ \/ __|
#   | |____ | || (_| |\__ \\__ \|  __/\__ \
#    \_____||_| \__,_||___/|___/ \___||___/
#

#    _  _       _                    ___  _
#   | || | ___ | | _ __  ___  _ _   / __|| | __ _  ___ ___ ___  ___
#   | __ |/ -_)| || '_ \/ -_)| '_| | (__ | |/ _` |(_-<(_-</ -_)(_-<
#   |_||_|\___||_|| .__/\___||_|    \___||_|\__,_|/__//__/\___|/__/
#                 |_|
class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


#    ___            ___  _           _
#   | _ \ ___  ___ | _ )| | ___  __ | |__
#   |   // -_)(_-< | _ \| |/ _ \/ _|| / /
#   |_|_\\___|/__/ |___/|_|\___/\__||_\_\
#
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


#    _  _                   _    ___  ___  ___
#   | \| |___ _  _ _ _ __ _| |  / _ \|   \| __|
#   | .` / -_) || | '_/ _` | | | (_) | |) | _|
#   |_|\_\___|\_,_|_| \__,_|_|  \___/|___/|___|
#
class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, odesolver, tol=1e-3):
        super(ODEBlock, self).__init__()
        self.odesolver = odesolver
        self.odefunc = odefunc
        self.tol = tol
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = self.odesolver(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
