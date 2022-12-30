import functools
import sys
sys.path.append('/home/park/workspace/PapersWithCode/Segmentation')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_utils import CustomDataset
from transformation import *

class FCN8s(nn.Module):

    def __init__(self, feature_extractor, n_class):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.n_class = n_class

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)


    def forward(self, x):
        extracted_feature = self.feature_extractor(x)
        x5 = extracted_feature['x5']
        x4 = extracted_feature['x4']
        x3 = extracted_feature['x3']

        h = self.relu(self.deconv1(x5))
        h = self.bn1(h + x4)

        h = self.relu(self.deconv2(h))
        h = self.bn2(h + x3)

        h = self.bn3(self.relu(self.deconv3(h)))
        h = self.bn4(self.relu(self.deconv4(h)))
        h = self.bn5(self.relu(self.deconv5(h)))
        output = self.classifier(h)

        return output

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


def test_model(model, data_loader):
    phase = 'valid'

    for data in data_loader[phase]:
        inputs = data['image']
        labels = data['label']
        names = data['name']
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        print(outputs.shape)
        break


if __name__ == '__main__':
    vgg16 = VGGNet('vgg16', 3)

    data_params = dict()
    data_params['root_dir'] = '/home/park/DATA/2022_11_23/trails'
    data_params['image_size'] = (128, 256)
    data_params['num_class'] = 14

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'valid': transforms.Compose([
            ConvertLabel(),
            ToTensor(),
            Rescale(data_params['image_size']),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    dataset = {mode:
        CustomDataset(params=data_params, mode=mode, transform=data_transforms[mode])
        for mode in ['valid']}
    dataset_sizes = {x: len(dataset[x]) for x in ['valid']}

    data_loader = {}

    data_loader['valid'] = DataLoader(dataset['valid'],
                            batch_size=16,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            # pin_memory_device=device,
                            prefetch_factor=2)

    vgg = VGGNet('vgg16', 3).to(device)
    fcn = FCN8s(vgg, 14).to(device)
    test_model(fcn, data_loader)

    # print(vgg16)
    # for param_tensor in fcn.state_dict():
    #     print(param_tensor, "\t", fcn.state_dict()[param_tensor].size())