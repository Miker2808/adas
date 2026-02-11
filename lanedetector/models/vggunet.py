import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.models import VGG16_BN_Weights

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class VGG_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1,pretrained=True
    ):
        super(VGG_UNET, self).__init__()

        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16_bn(weights=weights)

        # encoder is pretrained VGG with batch norm
        self.encoder1 = vgg.features[:6]   # [64, BN, ReLU, 64, BN, ReLU, maxpool]
        self.encoder2 = vgg.features[6:13]  # [128, BN, ReLU, 128, BN, ReLU, maxpool]
        self.encoder3 = vgg.features[13:23] # [256, BN, ReLU, 256, BN, ReLU, 256, BN, ReLU, maxpool]
        self.encoder4 = vgg.features[23:33] # [512, BN, ReLU, 512, BN, ReLU, 512, BN, ReLU, maxpool]
        self.encoder5 = vgg.features[33:43] # [512, BN, ReLU, 512, BN, ReLU, 512, BN, ReLU, maxpool]

        self.ups = nn.ModuleList()

        self.bottleneck = DoubleConv(512, 512)

        # Decoder
        self.ups = nn.ModuleList()

        self.ups.append(nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(1024, 512))  # 512(skip) + 512(up)
        
        self.ups.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(512, 256))  # 256(skip) + 256(up)
        
        self.ups.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(256, 128))  # 128(skip) + 128(up)
        
        self.ups.append(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(128, 64))  # 64(skip) + 64(up)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)
        
        # Decoder with skip connections
        skip_connections = [enc4, enc3, enc2, enc1]
        
        x = bottleneck
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

if __name__ == "__main__":
    x = torch.randn((3, 3, 161, 161))
    model = VGG_UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")