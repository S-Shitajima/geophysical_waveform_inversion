from timm import layers
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            kernel_size: int,
            padding: int,
            stride: int
        ) -> None:

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            layers.LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            layers.LayerNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    

class ResConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            kernel_size: int,
            stride: int,
            padding: int,
        ) -> None:

        super().__init__()
        self.residual = self._residual(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = layers.LayerNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = layers.LayerNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += r
        x = self.act2(x)
        return x
    
    def _residual(self, in_channels: int, out_channels: int) -> nn.Module:
        if in_channels == out_channels:
            return nn.Identity()
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class PreActResConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            kernel_size: int,
            stride: int,
            padding: int,
        ) -> None:

        super().__init__()
        self.residual = self._residual(in_channels, out_channels)
        self.norm1 = layers.LayerNorm2d(in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = layers.LayerNorm2d(out_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.residual(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x += r
        return x
    
    def _residual(self, in_channels: int, out_channels: int) -> nn.Module:
        if in_channels == out_channels:
            return nn.Identity()
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)
        

class PreActUpConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            kernel_size: int = 2,
            stride: int = 2
        ) -> None:

        super().__init__()
        self.up = nn.Sequential(
            layers.LayerNorm2d(in_channels),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            kernel_size: int = 2,
            stride: int = 2
        ) -> None:

        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            layers.LayerNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
    
