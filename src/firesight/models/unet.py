"""Configurable U-Net architecture for semantic segmentation."""

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolution layers with normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    """Encoder block: downsample and extract features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: upsample and combine skip features."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(x)

        # Handle odd image dimensions safely.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Four-level U-Net for multiclass semantic segmentation.

    Input:
        Tensor of shape [batch, channels, height, width].

    Output:
        Raw logits of shape [batch, num_classes, height, width].

    Apply softmax during inference, not inside the model.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        if in_channels < 1:
            raise ValueError("in_channels must be positive")

        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")

        c = base_channels

        # Encoder
        self.encoder1 = DoubleConv(in_channels, c)
        self.encoder2 = DownBlock(c, c * 2)
        self.encoder3 = DownBlock(c * 2, c * 4)
        self.encoder4 = DownBlock(c * 4, c * 8)

        # Bottleneck
        self.bottleneck = DownBlock(c * 8, c * 16)

        # Decoder
        self.decoder4 = UpBlock(c * 16, c * 8, c * 8)
        self.decoder3 = UpBlock(c * 8, c * 4, c * 4)
        self.decoder2 = UpBlock(c * 4, c * 2, c * 2)
        self.decoder1 = UpBlock(c * 2, c, c)

        # Pixel-wise classification head
        self.output_head = nn.Conv2d(
            c,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected a 4D input tensor [N, C, H, W]")

        if x.shape[1] != self.encoder1.layers[0].in_channels:
            raise ValueError(
                f"Expected {self.encoder1.layers[0].in_channels} "
                f"input channels, received {x.shape[1]}"
            )

        if x.shape[-2] < 16 or x.shape[-1] < 16:
            raise ValueError("Input height and width must be at least 16")

        if not x.is_floating_point():
            raise TypeError("Input image tensor must have a floating dtype")

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        return self.output_head(d1)
