import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, mode, in_channels):
        super().__init__()

        num_classes = 14
        self.in_channels = in_channels
        mode = mode.lower()
        self.cfg = self.make_cfgs()[mode]
        self.features, self.output_point = self.make_layers()

    def make_layers(self, batch_norm = False):
        layers = []
        output_point = {}

        in_channels = self.in_channels
        n_layer = 0
        for block_index, value_list in self.cfg.items():
            for value in value_list:
                if value == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    n_layer += 1
                else:
                    value = int(value)
                    conv2d = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
                        n_layer += 3
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                        n_layer += 2
                    in_channels = value
            output_point[block_index] = n_layer - 1

        return nn.Sequential(*layers), output_point


    def make_cfgs(self):
        cfgs = {}
        cfgs['vgg16'] = {
            1 : [64, 64, "M"],
            2 : [128, 128, "M"],
            3 : [256, 256, 256, "M"],
            4 : [512, 512, 512, "M"],
            5 : [512, 512, 512, "M"]}
        cfgs['vgg19'] = {
            1 : [64, 64, "M"],
            2 : [128, 128, "M"],
            3 : [256, 256, 256, 256, "M"],
            4 : [512, 512, 512, 512, "M"],
            5 : [512, 512, 512, 512, "M"]}
        return cfgs


    def forward(self, x):
        output = {}
        block_index = 1
        n_layers = len(self.features)
        for index in range(n_layers):
            x = self.features[index](x)

            if index == self.output_point[block_index]:
                output["x%d"%(block_index)] = x
                block_index += 1
        return output
