import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.models import ResNet18_Weights

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

class RESNET18_UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(RESNET18_UNET, self).__init__()
        self.name = "resnet18_unet"

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Encoder (ResNet18 channels: 64, 64, 128, 256, 512)
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        ) 
        self.maxpool = self.resnet.maxpool
        self.encoder2 = self.resnet.layer1 
        self.encoder3 = self.resnet.layer2 
        self.encoder4 = self.resnet.layer3 
        self.encoder5 = self.resnet.layer4 

        # Bottleneck
        self.bottleneck = DoubleConv(512, 512)

        # Decoder - Drastically reduced channel counts for speed
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec1 = DoubleConv(512, 256) # 256(skip) + 256(up)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128) 
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = DoubleConv(128, 64) 
        
        self.up4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec4 = DoubleConv(128, 64) 
        
        self.up5 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec5 = DoubleConv(32, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x) 
        p1 = self.maxpool(x1)
        x2 = self.encoder2(p1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        
        bottleneck = self.bottleneck(x5)
        
        # Decoder
        d1 = self.up1(bottleneck)
        # Use simple interpolation resizing if dimensions mismatch slightly due to padding
        if d1.shape != x4.shape: d1 = TF.resize(d1, size=x4.shape[2:])
        d1 = torch.cat((x4, d1), dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        if d2.shape != x3.shape: d2 = TF.resize(d2, size=x3.shape[2:])
        d2 = torch.cat((x3, d2), dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        if d3.shape != x2.shape: d3 = TF.resize(d3, size=x2.shape[2:])
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        if d4.shape != x1.shape: d4 = TF.resize(d4, size=x1.shape[2:])
        d4 = torch.cat((x1, d4), dim=1)
        d4 = self.dec4(d4)
        
        d5 = self.up5(d4)
        d5 = self.dec5(d5)
        
        return self.final_conv(d5)