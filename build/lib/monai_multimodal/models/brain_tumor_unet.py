from typing import Optional

import torch
from torch import nn
from monai.networks.nets import UNet


class BrainTumorUNet(nn.Module):
    """
    A configurable 3D U-Net suitable for brain tumor segmentation tasks, serving as a base multimodal model.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


