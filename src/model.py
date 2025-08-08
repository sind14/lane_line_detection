import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        return self.conv(inputs)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.conv_1 = DoubleConv(in_channels, 4)
        self.maxpool_1 = nn.MaxPool2d(2, 2)

        self.conv_2 = DoubleConv(4, 8)
        self.maxpool_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = DoubleConv(8, 16)

        self.upconv_2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv_up_2 = DoubleConv(16, 8)

        self.upconv_1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.conv_up_1 = DoubleConv(8, 4)

        self.final_conv = nn.Conv2d(4, out_channels, kernel_size=1)

    def forward(self, inputs):
        x1 = self.conv_1(inputs)
        x2 = self.conv_2(self.maxpool_1(x1))
        x3 = self.conv_3(self.maxpool_2(x2))

        x = self.upconv_2(x3)
        x = self.conv_up_2(torch.cat([x2, x], dim=1))

        x = self.upconv_1(x)
        x = self.conv_up_1(torch.cat([x1, x], dim=1))

        return self.final_conv(x)
