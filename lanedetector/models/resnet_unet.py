import torch
import torch.nn as nn
import torch.nn.functional as F
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

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class RESNET18_UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(RESNET18_UNET, self).__init__()
        self.name = "resnet18_attention_unet"

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.maxpool = self.resnet.maxpool
        self.encoder2 = self.resnet.layer1 
        self.encoder3 = self.resnet.layer2 
        self.encoder4 = self.resnet.layer3 
        self.encoder5 = self.resnet.layer4 

        self.bottleneck = DoubleConv(512, 512)

        # Attention Gates
        self.att4 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.att3 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.att2 = AttentionGate(F_g=128, F_l=64, F_int=32)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Decoder - Fixed Input Channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # dec1: 256 (att4) + 512 (up-sampled bottleneck) = 768
        self.dec1 = DoubleConv(768, 256) 
        
        # dec2: 128 (att3) + 256 (d1) = 384
        self.dec2 = DoubleConv(384, 128) 
        
        # dec3: 64 (att2) + 128 (d2) = 192
        self.dec3 = DoubleConv(192, 64) 
        
        # dec4: 64 (att1) + 64 (d3) = 128
        self.dec4 = DoubleConv(128, 64) 
        
        # Final upsample to match original image size
        self.dec5 = DoubleConv(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x) 
        p1 = self.maxpool(x1)
        x2 = self.encoder2(p1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        
        b = self.bottleneck(x5)
        
        # Decoder
        d1 = self.up(b)
        if d1.shape != x4.shape: d1 = TF.resize(d1, size=x4.shape[2:])
        s4 = self.att4(g=d1, x=x4)
        d1 = self.dec1(torch.cat((s4, d1), dim=1))
        
        d2 = self.up(d1)
        if d2.shape != x3.shape: d2 = TF.resize(d2, size=x3.shape[2:])
        s3 = self.att3(g=d2, x=x3)
        d2 = self.dec2(torch.cat((s3, d2), dim=1))
        
        d3 = self.up(d2)
        if d3.shape != x2.shape: d3 = TF.resize(d3, size=x2.shape[2:])
        s2 = self.att2(g=d3, x=x2)
        d3 = self.dec3(torch.cat((s2, d3), dim=1))
        
        d4 = self.up(d3)
        if d4.shape != x1.shape: d4 = TF.resize(d4, size=x1.shape[2:])
        s1 = self.att1(g=d4, x=x1)
        d4 = self.dec4(torch.cat((s1, d4), dim=1))
        
        d5 = self.up(d4)
        d5 = self.dec5(d5)
        
        return self.final_conv(d5)