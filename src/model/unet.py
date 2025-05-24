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
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Mish()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channles: int, out_channles: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channles,
            out_channels=out_channles,
            kernel_size=2,
            stride=2,
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Contracting path
        self.enc1 = ConvBlock(5, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(64, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(128, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512, 3, 1, 1)
        
        # Expanding path
        self.upconv3 = UpConvBlock(512, 256)
        self.dec3 = ConvBlock(512, 256, 3, 1, 1)

        self.upconv2 = UpConvBlock(256, 128)
        self.dec2 = ConvBlock(256, 128, 3, 1, 1)

        self.upconv1 = UpConvBlock(128, 64)
        self.dec1 = ConvBlock(128, 64, 3, 1, 1)
        
        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        
        # Classifier layer
        # self.clf = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(512*64*9, 1),
        # )
    
    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x1 = x # (N, 64, 72, 72)
        x = self.pool1(x)

        x = self.enc2(x)
        x2 = x # (N, 128, 36, 36)
        x = self.pool2(x)

        x = self.enc3(x)
        x3 = x # (N, 256, 18, 18)
        x = self.pool3(x)
        
        # Bottleneck
        x = self.bottleneck(x) # (N, 512, 9, 9)

        # Classifier
        # logit = self.clf(x)
        
        # Decoder
        x = self.upconv3(x) # (N, 256, 18, 18)
        x = torch.cat([x, x3], dim=1) # (N, 512, 18, 18)
        x = self.dec3(x) # (N, 256, 18, 18)

        x = self.upconv2(x) # (N, 128, 36, 36)
        x = torch.cat([x, x2], dim=1) # (N, 256, 36, 36)
        x = self.dec2(x) # (N, 128, 36, 36)

        x = self.upconv1(x) # (N, 64, 72, 72)
        x = torch.cat([x, x1], dim=1) # (N, 128, 72, 72)
        x = self.dec1(x) # (N, 64, 72, 72)

        # Final output
        logit = self.final(x)
        logit = logit[:, :, 1:71, 1:71]
        return logit