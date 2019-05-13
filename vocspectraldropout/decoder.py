import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from collections import OrderedDict


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


class ChannelAttention(nn.Module):
    """Channel attention module for the decoder.

    Since we take features from the feature detection network
    and features from the pyramid, we need to work out what features
    from the network to pay attention to based on the higher level features
    in the pyramid.

    This is done through an attention mechanism. We take the channels from
    the higher levels of the pyramid and the channels from the lower level features
    and concatenate them together. Then, we use AdaptiveAvgPool2d to collapse
    all channels into a vector of length n_lower_level_channels +
    n_pyramid_channels.

    We then run that through a mini perceptron that outputs a
    a vector of length n_lower_level_channels and take softmax. These can be
    used to re-weight the lower level features, so that we only pay attention
    to the ones that we care about (the idea is that the channel attention weight
    matrix learns which lower level channels correspond to the higher level
    channels that we care about based on the features present in the lower
    and upper levels).
    """

    def __init__(self, low_level_input_channels, pyramid_input_channels):
        """Initialize and create weights."""
        super().__init__()

        in_channels = low_level_input_channels + pyramid_input_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_channels, in_channels // 16)),
            ("relu1", nn.ReLU(inplace=False)),
            ("fc2", nn.Linear(in_channels // 16, low_level_input_channels)),
            ("sigmoid", nn.Sigmoid())
        ]))
        self.out_channels = low_level_input_channels

    def forward(self, x):
        """Compute channel attention."""
        batch_size, channels = x.shape[:2]

        # We reshape with additional width and height dimension
        # so that broadcasted multiplication works correctly
        x = self.avgpool(x).view(batch_size, channels)
        return self.net(x).view(batch_size, self.out_channels, 1, 1)


class Decoder(nn.Module):
    """Decode low level features and spatial pyramid pools."""

    def __init__(self,
                 low_level_input_channels,
                 low_level_output_channels,
                 pyramid_input_channels,
                 pyramid_output_channels,
                 num_classes,
                 use_channel_dropout=False,
                 use_channel_attention=False):
        """Initialize the decoder."""
        super().__init__()

        self.conv1 = ConvBNReLU(low_level_input_channels,
                                low_level_output_channels,
                                kernel_size=1,
                                bias=False)
        self.attention = ChannelAttention(low_level_input_channels,
                                          pyramid_input_channels) if use_channel_attention else None
        self.decoder_flow = nn.Sequential(
            ConvBNReLU(pyramid_input_channels + low_level_output_channels,
                       pyramid_output_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False),
            # TODO: Apply Channel Dropout if specified
            (nn.Dropout2d if use_channel_dropout else nn.Dropout)(0.5),
            ConvBNReLU(pyramid_output_channels,
                       pyramid_output_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False),
            (nn.Dropout2d if use_channel_dropout else nn.Dropout)(0.1),
            nn.Conv2d(pyramid_output_channels, num_classes, kernel_size=1, stride=1)
        )

        self._init_weight()

    def forward(self, x, low_level_features):
        # Interpolate the pyramid features to be the same size as
        # as the low_level features
        pyramid_features_width, pyramid_features_height = low_level_features.size()[2:]
        x = F.interpolate(x,
                          (pyramid_features_width,
                           pyramid_features_height),
                          mode='bilinear',
                          align_corners=True)

        if self.attention:
            attention = self.attention(torch.cat((x, low_level_features), dim=1))
            low_level_features = low_level_features * attention

        low_level_features = self.conv1(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)

        return self.decoder_flow(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
