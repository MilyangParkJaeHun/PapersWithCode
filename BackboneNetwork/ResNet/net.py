import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(
            in_channels=out_channels,
            out_channels=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if (self.in_channels != self.out_channels or self.stride != 1):
            self.downsample = nn.Sequential(
                conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Sequential()


    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)

        identity = self.downsample(x)

        h += identity
        h = self.relu(x)

        return h


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(
            in_channels=out_channels,
            out_channels=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(
            in_channels=out_channels,
            out_channels=out_channels * Bottleneck.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * Bottleneck.expansion)

        if (stride != 1 or
            in_channels != out_channels * Bottleneck.expansion):

            self.downsample = nn.Sequential(
                conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels * Bottleneck.expansion,
                    stride=stride),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion))
        else:
            self.downsample = nn.Sequential()


    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)

        h = self.conv3(h)
        h = self.bn3(h)

        shortcut = self.downsample(x)

        h += shortcut
        h = self.relu(h)

        return h


class ResNet(nn.Module):
    def __init__(self, mode, num_classes, start_channels=64):
        super().__init__()

        mode = mode.lower()
        block, repetitions = self.get_net_info(mode)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_channels = start_channels
        out_channels = start_channels
        self.conv2_x = self._make_layer(block, in_channels, out_channels, repetitions[0], 1)
        in_channels = out_channels * block.expansion
        out_channels *= 2
        self.conv3_x = self._make_layer(block, in_channels, out_channels, repetitions[1], 2)
        in_channels = out_channels * block.expansion
        out_channels *= 2
        self.conv4_x = self._make_layer(block, in_channels, out_channels, repetitions[2], 2)
        in_channels = out_channels * block.expansion
        out_channels *= 2
        self.conv5_x = self._make_layer(block, in_channels, out_channels, repetitions[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def get_net_info(self, mode):
        if mode == 'resnet18':
            return BasicBlock, (2, 2, 2, 2)
        if mode == 'resnet34':
            return BasicBlock, (3, 4, 6, 3)
        if mode == 'resnet50':
            return Bottleneck, (3, 4, 6, 3)
        if mode == 'resnet101':
            return Bottleneck, (3, 4, 23, 3)
        if mode == 'resnet152':
            return Bottleneck, (3, 8, 36, 3)


    def _make_layer(self, block, in_channels, out_channels, repetition, start_stride):
        layers = []

        layers.append(block(in_channels, out_channels, start_stride))
        for _ in range(1, repetition):
            in_channels = block.expansion * out_channels
            layers.append(block(in_channels, out_channels, 1))

        return nn.Sequential(*layers)


    def forward(self, x):
        h = self.conv1(x)
        h = self.max_pool1(h)

        h = self.conv2_x(h)
        h = self.conv3_x(h)
        h = self.conv4_x(h)
        h = self.conv5_x(h)

        h = self.avg_pool(h)
        h = torch.flatten(h, 1)
        h = self.fc(h)

        return h


if __name__ == '__main__':
    model = ResNet('resnet50', 14)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())