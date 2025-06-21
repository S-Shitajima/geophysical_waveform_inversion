from typing import Tuple, Union

from timm import layers
import torch
from torch import nn
import torch.nn.functional as F


class AdjustLayer(nn.Module):
    def __init__(self, in_height, in_width, out_height, out_width) -> None:
        super().__init__()
        self.diff_height = out_height - 2 * in_height
        self.diff_width = out_width - 2 * in_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.diff_height == 0 and self.diff_width == 0:
            return x
        
        elif self.diff_height > 0 or self.diff_width > 0:
            left = self.diff_width // 2
            right = self.diff_width - left
            top = self.diff_height // 2
            bottom = self.diff_height - top
            return F.pad(x, pad=(left, right, top, bottom), mode="reflect")
        
        elif self.diff_height < 0 or self.diff_width < 0:
            left = -self.diff_width // 2
            right = -self.diff_width - left
            top = -self.diff_height // 2
            bottom = -self.diff_height - top
            return x[:, :, top: x.shape[2]-bottom, left: x.shape[3]-right]
        

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
        
    
class ResConvSCSEBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
        ) -> None:

        super().__init__()
        self.residual = self._residual(in_channels, out_channels, stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = layers.LayerNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.norm2 = layers.LayerNorm2d(out_channels)
        self.scse = SCSEBlock(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.scse(x)
        x += r
        x = self.act2(x)
        return x
    
    def _residual(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        if in_channels == out_channels:
            return nn.Identity()
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)


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
    

class UpPixelShuffleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            upscale_factor: int = 2,
        ) -> None:

        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*(upscale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            layers.LayerNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
    

class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel SE
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # Spatial SE
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        cse = self.cSE(x) * x
        sse = self.sSE(x) * x
        return cse + sse


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        x_cat = self.conv(x_cat)
        attn = torch.sigmoid(x_cat)
        return x * attn # (B, C, H, W)