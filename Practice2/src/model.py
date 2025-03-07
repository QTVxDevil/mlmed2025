import torch
import torch.nn as nn
import torch.nn.functional as F

class ENet(nn.Module):
    def __init__(self, num_classes=1):
        super(ENet, self).__init__()
        
        self.initial_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.encoder1 = self.conv_block(16, 64)
        self.encoder2 = self.conv_block(64, 128)

        self.decoder1 = self.conv_block(128, 64)
        self.decoder2 = self.conv_block(64, 16)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.initial_block(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = ENet()
    print(model)
        