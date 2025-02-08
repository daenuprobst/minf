from typing import Tuple
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple,
        stride: int | Tuple = 1,
        padding: int | Tuple = 1,
    ):
        super().__init__()
        self.in_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn_input = nn.BatchNorm3d(in_channels * 2)

        self.layer_1 = nn.Conv3d(
            in_channels=in_channels * 2,
            out_channels=in_channels * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn_1 = nn.BatchNorm3d(in_channels * 4)

        self.out_layer = nn.Conv3d(
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn_output = nn.BatchNorm3d(out_channels)

        self.pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.in_layer(x)))
        x = F.relu(self.bn_1(self.layer_1(x)))
        x = F.relu(self.bn_output(self.out_layer(x)))
        x = self.pooling(x)
        return x.flatten(1)
