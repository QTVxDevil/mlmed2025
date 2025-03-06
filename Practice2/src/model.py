import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
        self.encoder1 = conv_block(1, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder2 = conv_block(64, 128)
        
        self.decoder2 = conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.pool(x1)
        x3 = self.encoder2(x2)
        x4 = self.pool(x3)
        
        x5 = self.decoder2(x4)
        x6 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=True)
        x7 = self.final_conv(x6)
        
        x8 = F.interpolate(x7, size=(256, 256), mode="bilinear", align_corners=True)
        
        return torch.sigmoid(x8)
        