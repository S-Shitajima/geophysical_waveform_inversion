from timm import create_model
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            stride: int
        ) -> None:

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.LayerNorm(out_channels),
            nn.Mish(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.LayerNorm(out_channels),
            nn.Mish()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(
            self,
            in_channles: int,
            out_channles: int,
            kernel_size: int = 2,
            stride: int = 2
        ) -> None:

        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channles,
            out_channels=out_channles,
            kernel_size=kernel_size,
            stride=stride,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Image2ImageUNet(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = create_model(
            model_name=model_name,
            pretrained=True,
            in_chans=5,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        chs = []
        for i, ch in enumerate(self.encoder(torch.randn(1, 5, 288, 288))):
            print(i, ch.shape)
            chs.append(ch.shape[1])
        chs = chs[::-1]
        
        self.convs = nn.ModuleDict()
        for i, (ch1, ch2) in enumerate(zip(chs[:-1], chs[1:])):
            self.convs[str(i)] = nn.ModuleList([UpConvBlock(ch1, ch2), ConvBlock(2*ch2, ch2, 3, 1, 1)])
        
        self.final = nn.Conv2d(chs[3], 1, kernel_size=1, stride=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.encoder(x)
        x = xs[-1] # (N, C4, 9, 9)
        for i in range(len(self.convs)):
            x = self.convs[str(i)][0](x)
            x = torch.cat([x, xs[len(self.convs)-1-i]], dim=1)
            x = self.convs[str(i)][1](x)
        
        logit = self.final(x) # (N, 1, 72, 72)
        logit = logit[:, :, 1:71, 1:71]
        return logit