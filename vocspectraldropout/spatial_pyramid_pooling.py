import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ConvBNReLU(nn.Module):
    """A simple building block that does a convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels, out_channels, **kwargs):
        """Initialize layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self._init_weight()

    def forward(self, x):
        return self.net(x)


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SpatialPoolingPyramid(nn.Module):
    """Pool over many different convolutions with different dilations."""

    def __init__(self, input_channels, dilations, output_pooling_channels=256):
        """Initialize the differnent pooling layers."""
        super().__init__()
        self.pooling_layers = nn.ModuleList([
            ConvBNReLU(input_channels, output_pooling_channels, kernel_size=1, bias=False, padding=0)
        ] + [
            ConvBNReLU(input_channels, output_pooling_channels, kernel_size=3, bias=False, padding=d, dilation=d)
            for d in dilations
        ])
        self.global_average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNReLU(input_channels, output_pooling_channels, kernel_size=1, bias=False, padding=0)
        )
        self.concat_conv = ConvBNReLU(output_pooling_channels * (len(dilations) + 2),
                                      output_pooling_channels,
                                      kernel_size=1)
        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def forward(self, x):
        """Forward through layers.

        Take the input through each the pooling layers, then concatenate them
        with the maxpooled layers (after a conv such that all pooling layers
        have the same number of channels.

        Note that the image size after global average pooling may not be the same
        as the image size after pyramid pooling. Therefore, we need to interpolate
        (upscale or downscale) the GAP'd image to be the same dimension as the
        pyramid pooled images.
        """
        pyramid_pooling_outputs = [pooling_layer(x) for pooling_layer in self.pooling_layers]
        global_average_pooling_output = self.global_average_pooling(x)

        pyramid_pooling_output_width, pyramid_pooling_output_height = pyramid_pooling_outputs[-1].size()[2:]

        # Upscale or downscale the global average pooling output
        global_average_pooling_output = F.interpolate(global_average_pooling_output,
                                                      size=(pyramid_pooling_output_width,
                                                            pyramid_pooling_output_height),
                                                      mode='bilinear',
                                                      align_corners=True)

        # Concatenate along the channels. We now have (len(pyramid_pooling_outputs) + 1) * out_channels channels
        concatenated_pooling_outputs = torch.cat(pyramid_pooling_outputs + [global_average_pooling_output], dim=1)

        # Convolution filter to reduce the large number of concatenated channels into
        # just output_channels
        outputs = self.concat_conv(concatenated_pooling_outputs)

        # Apply regular dropout to the outputs here
        return self.dropout(outputs)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
