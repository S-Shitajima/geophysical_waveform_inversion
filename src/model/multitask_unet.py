from timm import create_model, layers
import torch
from torch import nn
import torch.nn.functional as F

from .unet_layers import (
    AdjustLayer,
    ConvBlock,
    ResConvBlock,
    ResConvSCSEBlock,
    PreActResConvBlock,
    PreActUpConvBlock,
    SCSEBlock,
    SpatialAttention,
    UpConvBlock,
    UpPixelShuffleBlock,
)


class MultiTaskUNet(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            pretrained: bool,
            height: int,
            width: int,
        ) -> None:
        super().__init__()

        # encoder
        self.model_name = model_name
        if "swin" in model_name:
            self.encoder = create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=5,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                img_size=(height, width),
            )
        else:
            self.encoder = create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=5,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        
        shapes = []
        for i, out in enumerate(self.encoder(torch.randn(1, 5, height, width))):
            print(i, out.shape)
            if "swin" in model_name:
                _, h, w, c = out.shape
            else:
                _, c, h, w = out.shape
            shapes.append((c, h, w))
        shapes = shapes[::-1]
        print(shapes)
        
        # decorder
        self.convs = nn.ModuleDict()
        for i, (sh1, sh2) in enumerate(zip(shapes[:-1], shapes[1:])):
            ch1, h1, w1 = sh1
            ch2, h2, w2 = sh2
            self.convs[str(i)] = nn.ModuleList(
                [
                    # UpConvBlock(ch1, ch2, 2, 2),
                    UpPixelShuffleBlock(ch1, ch2, 2),
                    # AdjustLayer(h1, w1, h2, w2),
                    # ResConvSCSEBlock(2*ch2, ch2, 3, 1, 1),
                    ResConvBlock(2*ch2, ch2, 3, 1, 1),
                ]
            )
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(shapes[-1][0], 1, kernel_size=1, bias=False),
        )
        
        # Classifier layer
        self.global_pool = nn.Sequential(
            # SpatialAttention(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1),
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.clf = nn.Linear(shapes[0][0], num_classes, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.encoder(x)
        if "swin" in self.model_name:
            xs = [x.permute(0, 3, 1, 2) for x in xs]
        x = xs[-1]

        # classification
        x_clf = self.global_pool(x)
        clf_logit = sum([self.clf(dropout(x_clf)) for dropout in self.dropouts]) / 5

        # decoding
        for i in range(len(self.convs)):
            x = self.convs[str(i)][0](x)
            # x = self.convs[str(i)][1](x)
            x = torch.cat([x, xs[len(self.convs)-1-i]], dim=1)
            x = self.convs[str(i)][1](x)
        
        # final output
        reg_logit = self.final(x)
        reg_logit = reg_logit[:, :, 1:71, 1:71]
        return reg_logit, clf_logit