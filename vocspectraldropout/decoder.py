import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ConvBNReLU(nn.Module):
    """Convolution, BatchNorm, ReLU"""

    def __init__(self, in_channels, out_channels, **kwargs):
        """Initialize."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Decode low level features and spatial pyramid pools."""

    def __init__(self,
                 low_level_input_channels,
                 low_level_output_channels,
                 pyramid_input_channels,
                 pyramid_output_channels,
                 num_classes):
        """Initialize the decoder."""
        super().__init__()

        self.conv1 = ConvBNReLU(low_level_input_channels,
                                low_level_output_channels,
                                kernel_size=1,
                                bias=False)
        self.decoder_flow = nn.Sequential(
            ConvBNReLU(pyramid_input_channels + low_level_output_channels,
                       pyramid_output_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False),
            nn.Dropout(0.5),
            ConvBNReLU(pyramid_output_channels,
                       pyramid_output_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False),
            nn.Dropout(0.1),
            nn.Conv2d(pyramid_output_channels, num_classes, kernel_size=1, stride=1)
        )

        self._init_weight()

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)

        # Interpolate the pyramid features to be the same size as
        # as the low_level features
        pyramid_features_width, pyramid_features_height = low_level_features.size()[2:]

        x = F.interpolate(x,
                          (pyramid_features_width,
                           pyramid_features_height),
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)

        return self.decoder_flow(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
