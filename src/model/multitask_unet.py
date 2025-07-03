from typing import Tuple

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

        # self._update_stem(model_name)

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
        self.convs = nn.ModuleList()
        for i, (sh1, sh2) in enumerate(zip(shapes[:-1], shapes[1:])):
            ch1, h1, w1 = sh1
            ch2, h2, w2 = sh2
            self.convs.append(nn.ModuleList(
                [
                    UpConvBlock(ch1, ch2, 2, 2),
                    # UpPixelShuffleBlock(ch1, ch2, 2),
                    # AdjustLayer(h1, w1, h2, w2),
                    # ResConvSCSEBlock(2*ch2, ch2, 3, 1, 1),
                    ResConvBlock(2*ch2, ch2, 3, 1, 1),
                ]
            ))
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(shapes[-1][0], shapes[-1][0], kernel_size=3, stride=1),
            layers.LayerNorm2d(shapes[-1][0], eps=1e-05),
            nn.GELU(),
            nn.Conv2d(shapes[-1][0], 1, kernel_size=1, stride=1, bias=False),
        )
        
        # Classifier layer
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1),
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.clf = nn.Linear(shapes[0][0], num_classes, bias=False)

        print(self.encoder)
    
    # def _update_stem(self, model_name):
        # if "caformer" in model_name:
        #     m = self.encoder
        #     m.stem.conv.stride = (4, 1)
        #     m.stem.conv.padding = (0, 4)
        #     m.stem = nn.Sequential(
        #         nn.ReflectionPad2d((0, 0, 78, 78)),
        #         m.stem,
        #     )
        #     m.stages_0.downsample = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

        # elif "convnext" in model_name:
        #     m = self.encoder
        #     m.stem.conv.stride = (4, 1)
        #     m.stem.conv.padding = (0, 2)
        #     m.stages_0.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 2), stride=(4, 1)),
        #         layers.LayerNorm2d(128, eps=1e-06),
        #         nn.GELU(),
        #     )

        # elif "focalnet_base_lrf" in model_name:
        #     m = self.encoder
        #     m.stem.proj.stride = (4, 1)
        #     m.stem.proj.padding = (0, 2)
        #     m.layers_0.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 2), stride=(4, 1)),
        #         layers.LayerNorm2d(128, eps=1e-05),
        #         nn.GELU(),
        #     )

        # elif "focalnet_large_fl4" in model_name:
        #     m = self.encoder
        #     m.stem.proj.stride = (4, 1)
        #     m.stem.proj.padding = (0, 4)
        #     m.stem = nn.Sequential(
        #         nn.ReflectionPad2d((0, 0, 78, 78)),
        #         m.stem,
        #     )
        #     m.layers_0.downsample = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.encoder(x)
        if "swin" in self.model_name:
            xs = [x.permute(0, 3, 1, 2) for x in xs]
        x = xs[-1]

        # classification
        x_clf = self.global_pool(x)
        clf_logit = torch.mean(torch.stack([self.clf(dropout(x_clf)) for dropout in self.dropouts]), dim=0)

        # decoding
        for i, conv12 in enumerate(self.convs):
            x = conv12[0](x)
            x = torch.cat([x, xs[len(self.convs)-1-i]], dim=1)
            x = conv12[1](x)
        
        # final output
        reg_logit = self.final(x)
        # reg_logit = reg_logit[:, :, 1:71, 1:71]
        return reg_logit, clf_logit
    