import torch.nn as nn
import math
import torch


from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableQuantizableConv2d, SlimmableConv2d, SlimmableLinear
from utils.config import FLAGS

class BasicBlock(nn.Module):
    def __init__(self, inp, outp, stride):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]

        self.conv1 = SlimmableQuantizableConv2d(inp, outp, 3, stride, 1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(outp, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SlimmableQuantizableConv2d(outp, outp, 3, 1, 1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(outp, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list))

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.downsample = nn.Sequential(
                SlimmableQuantizableConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchableBatchNorm2d(outp, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list)),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.residual_connection:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        midp = [i//4 for i in outp]
        layers = [
            SlimmableQuantizableConv2d(inp, midp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(midp, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list)),
            nn.ReLU(inplace=True),

            SlimmableQuantizableConv2d(midp, midp, 3, stride, 1, bias=False),
            SwitchableBatchNorm2d(midp, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list)),
            nn.ReLU(inplace=True),

            SlimmableQuantizableConv2d(midp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list)),
        ]
        self.body = nn.Sequential(*layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                SlimmableQuantizableConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchableBatchNorm2d(outp, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list)),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        self.features = []

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]
        feats = [64, 128, 256, 512]
        channels = [
            int(64 * width_mult) for width_mult in FLAGS.width_mult_list]

        if num_classes == 200:
            self.conv1 = SlimmableConv2d([3 for _ in range(len(channels))], channels, 3, 1, 1, bias=False)
        else:
            self.conv1 = SlimmableConv2d([3 for _ in range(len(channels))], channels, 7, 2, 3, bias=False)

        self.bn1 = SwitchableBatchNorm2d(channels, len(FLAGS.bitwidth_list), len(FLAGS.bitactiv_list))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # body
        expansion = 1 if (FLAGS.depth == 18) or (FLAGS.depth == 34) else 4

        for stage_id, n in enumerate(self.block_setting):
            outp = [
                int(feats[stage_id]*width_mult*expansion)
                for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                if i == 0 and stage_id != 0:
                    if (FLAGS.depth == 18) or (FLAGS.depth == 34):
                        self.features.append(BasicBlock(channels, outp, 2))
                    else:
                        self.features.append(Block(channels, outp, 2))
                else:
                    if (FLAGS.depth == 18) or (FLAGS.depth == 34):
                        self.features.append(BasicBlock(channels, outp, 1))
                    else:
                        self.features.append(Block(channels, outp, 1))
                channels = outp

        self.layer1 = [self.features[0], self.features[1]]
        self.layer2 = [self.features[2], self.features[3]]
        self.layer3 = [self.features[4], self.features[5]]
        self.layer4 = [self.features[6], self.features[7]]

        # make it nn.Sequential
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)
        self.layer4 = nn.Sequential(*self.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        # classifier
        self.outp = channels
        self.fc = SlimmableLinear(self.outp, [num_classes for _ in range(len(self.outp))])
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
