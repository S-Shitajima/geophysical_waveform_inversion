from timm import create_model, layers
import torch
from torch import nn

from .unet_layers import ConvBlock, ResConvBlock, PreActResConvBlock, PreActUpConvBlock, UpConvBlock


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
        
        # Change the first conv layer’s kernel size and stride from 4×4 to 2×2
        # if "nextvit" in model_name:
        #     stem2_out_channels = self.encoder["stem_2"].conv.out_channels
        #     stem3_out_channels = self.encoder["stem_3"].conv.out_channels
        #     self.encoder["stem_3"].conv = nn.Conv2d(in_channels=stem2_out_channels, out_channels=stem3_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # elif "convnext" in model_name:
        #     stem0_out_channels = self.encoder["stem_0"].out_channels
        #     self.encoder["stem_0"] = nn.Conv2d(in_channels=5, out_channels=stem0_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # print(self.encoder)
        
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
            self.convs[str(i)] = nn.ModuleList([UpConvBlock(ch1, ch2, h2, w2),
                                                ResConvBlock(2*ch2, ch2, h2, w2, 3, 1, 1)])
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(shapes[-1][0], shapes[-1][0], kernel_size=3, stride=1),
            layers.LayerNorm2d(shapes[-1][0]),
            nn.GELU(),
            nn.Conv2d(shapes[-1][0], 1, kernel_size=1, stride=1),
        )
        
        # Classifier layer
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(shapes[0][0], num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.encoder(x)
        if "swin" in self.model_name:
            xs = [x.permute(0, 3, 1, 2) for x in xs]
        x = xs[-1] # (N, C4, 9, 9)
        clf_logit = self.clf(x)

        for i in range(len(self.convs)):
            x = self.convs[str(i)][0](x)
            x = torch.cat([x, xs[len(self.convs)-1-i]], dim=1)
            x = self.convs[str(i)][1](x)
        
        # Final output
        logit = self.final(x)
        # logit = logit[:, :, 1:71, 1:71]
        return logit, clf_logit