# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, use_sigmoid=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.use_sigmoid = use_sigmoid

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 100, 128, 200, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            if self.use_sigmoid:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            else:
                self.convs[("dispconv", s)] = nn.Sequential(
                    ConvBlock(self.num_ch_dec[s], self.num_ch_dec[s]),
                    nn.Conv2d(self.num_ch_dec[s], self.num_output_channels, 1))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if self.use_skips and i > 0:
                x = [input_features[i - 1], 
                     F.interpolate(x, size=input_features[i - 1].shape[2:], mode="nearest")]
            else:
                x = [upsample(x)]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if self.use_sigmoid:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                else:
                    self.outputs[("disp", i)] = self.convs[("dispconv", i)](x)

        return self.outputs

class DebugDepthDecoder(DepthDecoder):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, use_sigmoid=True):
        super(DebugDepthDecoder, self).__init__(num_ch_enc, scales, num_output_channels, use_skips, use_sigmoid)

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x1 = self.convs[("upconv", i, 0)](x)
            if self.use_skips and i > 0:
                x2 = [input_features[i - 1], 
                     F.interpolate(x1, size=input_features[i - 1].shape[2:], mode="nearest")]
            else:
                x2 = [upsample(x1)]
            x3 = torch.cat(x2, 1)
            x4 = self.convs[("upconv", i, 1)](x3)

            self.outputs[("debug_0", i)] = x
            self.outputs[("debug_1", i)] = x1
            self.outputs[("debug_2", i)] = x3
            self.outputs[("debug_4", i)] = x4

            if i in self.scales:
                if self.use_sigmoid:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x4))
                else:
                    self.outputs[("disp", i)] = self.convs[("dispconv", i)](x4)
            x = x4
        return self.outputs