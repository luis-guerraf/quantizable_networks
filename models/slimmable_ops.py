import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantize_op import DoReFa_A, DoReFa_W
from utils.config import FLAGS


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list, num_bitwidths, num_bitactiv):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for _ in range(num_bitwidths):
            for _ in range(num_bitactiv):
                for i in num_features_list:
                    bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True
        self.bitwidth = max(FLAGS.bitwidth_list)
        self.bitactiv = max(FLAGS.bitactiv_list)
        self.num_widths = len(FLAGS.width_mult_list)
        self.chunks = len(FLAGS.width_mult_list)*len(FLAGS.bitwidth_list)

    def forward(self, input):
        idx_b = FLAGS.bitwidth_list.index(self.bitwidth)
        idx_a = FLAGS.bitactiv_list.index(self.bitactiv)
        idx_w = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[self.chunks*idx_a + self.num_widths*idx_b + idx_w](input)
        return y


class SlimmableQuantizableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableQuantizableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)
        self.bitwidth = max(FLAGS.bitwidth_list)
        self.bitactiv = max(FLAGS.bitactiv_list)

    def forward(self, input):
        # Quantizable settings (quantize before slim)
        idx_b = FLAGS.bitwidth_list.index(self.bitwidth)
        idx_a = FLAGS.bitactiv_list.index(self.bitactiv)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # Activation Quantization
        if (input.size(1) != 3):
            if FLAGS.bitactiv_list[idx_a] != 32:
                input = DoReFa_A(input, FLAGS.bitactiv_list[idx_a])

        # Weight Quantization
        if FLAGS.bitwidth_list[idx_b] != 32:
            self.weight.data = DoReFa_W(self.weight.org, FLAGS.bitwidth_list[idx_b])
        else:
            # They must be copied, otherwise they'll use the previous quantized
            self.weight.data.copy_(self.weight.org)

        # Slimmable settings
        idx_w = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx_w]
        self.out_channels = self.out_channels_list[idx_w]
        self.groups = self.groups_list[idx_w]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]

        # Forward
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableQuantizableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableQuantizableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)
        self.bitwidth = max(FLAGS.bitwidth_list)
        self.bitactiv = max(FLAGS.bitactiv_list)

    def forward(self, input):
        # Quantizable settings (quantize before slim)
        idx_b = FLAGS.bitwidth_list.index(self.bitwidth)
        idx_a = FLAGS.bitactiv_list.index(self.bitactiv)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # Activation function
        if FLAGS.bitactiv_list[idx_a] != 32:
            input = DoReFa_A(input, FLAGS.bitactiv_list[idx_a])

        if FLAGS.bitwidth_list[idx_b] != 32:
            self.weight.data = DoReFa_W(self.weight.org, FLAGS.bitwidth_list[idx_b])
        else:
            # They must be copied, otherwise they'll use the previous quantized
            self.weight.data.copy_(self.weight.org)

        # Slimmable settings
        idx_w = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx_w]
        self.out_features = self.out_features_list[idx_w]
        weight = self.weight[:self.out_features, :self.in_features]

        # Forward
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


