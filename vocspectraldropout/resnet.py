import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from collections import OrderedDict, defaultdict

import math


class Block(nn.Module):
    """A block in a ResNet."""

    CHANNEL_EXPANSION = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 drop_rate=0.0):
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            # 1x1 Convolution (rescale image activations)
            ("conv1", nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                bias=False)),
            ("bn1", nn.BatchNorm2d(out_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            # Potentially downsampling 3x3 convolution with dilation
            ("conv2", nn.Conv2d(out_channels,
                                out_channels,
                                kernel_size=3,
                                stride=stride,
                                dilation=dilation,
                                padding=dilation,
                                bias=False)),
            ("bn2", nn.BatchNorm2d(out_channels)),
            ("relu2", nn.ReLU(inplace=False)),
            #TODO: Added a channel dropout layer here
            ("dropout2",nn.Dropout2d(p=drop_rate)),
            # 1x1 Convlution to increase the number of planes
            ("conv3", nn.Conv2d(out_channels,
                                out_channels * Block.CHANNEL_EXPANSION,
                                kernel_size=1,
                                bias=False)),
            ("bn3", nn.BatchNorm2d(out_channels * Block.CHANNEL_EXPANSION)),
        ]))

        if stride != 1 or in_channels != out_channels * Block.CHANNEL_EXPANSION:
            # Skip connection, no need for activation
            self.skip = nn.Sequential(*[
                nn.Conv2d(in_channels,
                          out_channels * Block.CHANNEL_EXPANSION,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * Block.CHANNEL_EXPANSION)
            ])
        else:
            self.skip = lambda x: x

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class BlockGroup(nn.Module):
    """A group of blocks with the same stride and channels."""

    def __init__(self,
                 input_channels,
                 output_channels,
                 layers,
                 stride=1,
                 dilation=1,
                 drop_rate=0.0):
        """Initialize the group of blocks."""
        super().__init__()

        self.net = nn.Sequential(*([
            # First, go from input_channels to output_channels * CHANNEL_EXPANSION
            Block(input_channels,
                  output_channels,
                  stride=stride,
                  dilation=dilation,
                  drop_rate=drop_rate)
        ] + [
            # Now, go from output_channels * CHANNEL_EXPANSION to
            # output_channels * CHANNEL_EXPANSION, but with no
            # downsampling via stride
            Block(output_channels * Block.CHANNEL_EXPANSION,
                  output_channels,
                  dilation=dilation,
                  drop_rate=drop_rate)
            for i in range(1, layers)
        ]))

    def forward(self, x):
        return self.net(x)



class ExpandingBlockGroup(nn.Module):
    """A group of blocks where convolutions are dilated on each layer."""

    def __init__(self,
                 input_channels,
                 output_channels,
                 blocks,
                 stride=1,
                 dilation=1,
                 drop_rate=0.0):
        """Initialize the group of blocks."""
        super().__init__()

        self.net = nn.Sequential(*([
            # First, go from input_channels to output_channels * CHANNEL_EXPANSION
            Block(input_channels,
                  output_channels,
                  stride=stride,
                  dilation=dilation,
                  drop_rate=drop_rate)
        ] + [
            # Now, go from output_channels * CHANNEL_EXPANSION to
            # output_channels * CHANNEL_EXPANSION, but with no
            # downsampling via stride
            Block(output_channels * Block.CHANNEL_EXPANSION,
                  output_channels,
                  dilation=dilation * blocks[i],
                  drop_rate=drop_rate)
            for i in range(1, len(blocks))
        ]))

    def forward(self, x):
        return self.net(x)


FILTER_KEYS = ["low_level_net", "net"]
RENAME_KEYS = defaultdict(lambda: None,
                          low_level_block="layer1",
                          skip="downsample")


def rename_key(key):
    if key in RENAME_KEYS:
        return RENAME_KEYS[key]

    return key

def filter_key(key):
    path = key.split(".")
    filtered = ([rename_key(k) for k in path if k not in FILTER_KEYS])
    return ".".join(filtered)


class ResNet(nn.Module):
    """A group of groups of blocks."""

    def __init__(self,
                 input_channels,
                 block_layers,
                 load_pretrained=None,
                 drop_rate=0.0):
        """Initialize the resnet."""
        super().__init__()

        start_channels = 64
        self.low_level_net = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(input_channels,
                                start_channels,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False)),
            ("bn1", nn.BatchNorm2d(start_channels)),
            ("relu", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ("low_level_block", BlockGroup(start_channels, start_channels, block_layers[0], stride=1, dilation=1, drop_rate=drop_rate))
        ]))
        self.net = nn.Sequential(OrderedDict(([
            ("layer{}".format(i + 1), BlockGroup(start_channels * (2 ** (i - 1)) * Block.CHANNEL_EXPANSION,
                                                 start_channels * (2 ** i),
                                                 block_layers[i],
                                                 dilation=1,
                                                 stride=2,
                                                 drop_rate=drop_rate))
            for i in range(1, len(block_layers))
        ] + [
            ("layer{}".format(len(block_layers) + 1),
             ExpandingBlockGroup(start_channels * (2 * (len(block_layers) - 1) * Block.CHANNEL_EXPANSION),
                                 start_channels * (2 ** len(block_layers)),
                                 [2 ** i for i in range(len(block_layers))],
                                 stride=1,
                                 dilation=2,
                                 drop_rate=drop_rate))
        ])))

        self.initialize_weights()

        if load_pretrained:
            self.load_weights_from_pretrained(load_pretrained)

    def forward(self, x):
        low_level_features = self.low_level_net(x)
        return self.net(low_level_features), low_level_features


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_weights_from_pretrained(self, url):
        pretrained_dict = model_zoo.load_url(url)
        new_state_dict = {}
        consumed_pretrain_keys = set([])
        for key in self.state_dict():
            if "num_batches_tracked" in key:
                continue

            pretrain_key = filter_key(key)
            if pretrain_key not in pretrained_dict:
                raise RuntimeError("Expected {} to be in pretrained_dict, but was not.".format(
                    pretrain_key
                ))

            print("{} -> {}".format(pretrain_key, key))
            new_state_dict[key] = pretrained_dict[pretrain_key]
            consumed_pretrain_keys |= set([pretrain_key])

        print("Remaining unconsumed keys: {}".format([
            k for k in pretrained_dict if k not in consumed_pretrain_keys
        ]))
        print("Keys in state dict that have not been set:\n - {}\n".format("\n - ".join([
            k for k in self.state_dict() if k not in new_state_dict
        ])))
        self.load_state_dict(new_state_dict)


def build_backbone(input_channels, drop_rate=0.0):
    return ResNet(input_channels,
                  [3, 4, 23],
                  load_pretrained="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
                  drop_rate=drop_rate)

