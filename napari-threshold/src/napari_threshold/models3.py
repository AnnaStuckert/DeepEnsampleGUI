import torch
import torch.nn as nn


class UNet2(nn.Module):
    def __init__(self, in_f=1, out_f=1, dropout_rate=0.7):
        super(UNet2, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_f, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        # Final Output
        self.final_conv = nn.Conv2d(64, out_f, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        x = self.pool(enc1)
        x = self.dropout(x)

        enc2 = self.enc_conv2(x)
        x = self.pool(enc2)
        x = self.dropout(x)

        enc3 = self.enc_conv3(x)
        x = self.pool(enc3)
        x = self.dropout(x)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec_conv3(x)

        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec_conv2(x)

        x = self.upsample(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec_conv1(x)

        # Final Output
        x = self.final_conv(x)
        return x
