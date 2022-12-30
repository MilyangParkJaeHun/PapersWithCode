import os
import sys
sys.path.append(os.environ['WORKSPACE_PATH'])

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from BackboneNetwork.VGGNet.net import VGGNet
from Segmentation.data_utils import CustomDataset
from Segmentation.transformation import *


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

    vgg = VGGNet('vgg19', 3).to(device)
    fcn = FCN8s(vgg, 14).to(device)
    test_model(fcn, data_loader)