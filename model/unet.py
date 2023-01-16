import torch as th
from torch.utils import checkpoint
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, resnet=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.resnet=resnet
        if self.resnet:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        if self.resnet:
            return self.double_conv(x) + self.skip_connection(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,resnet=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,resnet=resnet)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, resnet=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, resnet=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = th.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=64, bilinear=False,resnet=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_filters = n_filters
        self.resnet = resnet
        factor = 2 if self.bilinear else 1

        self.input = (DoubleConv(self.n_channels, self.n_filters*1, resnet=self.resnet))
        self.down1 = (Down(self.n_filters*1, self.n_filters*2, self.resnet))
        self.down2 = (Down(self.n_filters*2, self.n_filters*4, self.resnet))
        self.down3 = (Down(self.n_filters*4, self.n_filters*8, self.resnet))
        self.down4 = (Down(self.n_filters*8, (self.n_filters*16) // factor, self.resnet))

        self.up1 = (Up(self.n_filters*16, (self.n_filters*8) // factor, self.bilinear))
        self.up2 = (Up(self.n_filters*8, (self.n_filters*4) // factor, self.bilinear))
        self.up3 = (Up(self.n_filters*4, (self.n_filters*2) // factor, self.bilinear))
        self.up4 = (Up(self.n_filters*2, self.n_filters*1, self.bilinear))
        self.out = (OutConv(self.n_filters, self.n_classes))

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits

    def use_checkpointing(self):
        self.input = checkpoint(self.input)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.up1 = checkpoint(self.up1)
        self.up2 = checkpoint(self.up2)
        self.up3 = checkpoint(self.up3)
        self.up4 = checkpoint(self.up4)
        self.out = checkpoint(self.out)