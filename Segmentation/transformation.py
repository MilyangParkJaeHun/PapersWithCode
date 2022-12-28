import cv2
import torch
import numpy as np
from skimage import io, transform
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class ConvertLabel(object):

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        lbl = cv2.split(label)[0]

        return {'image': image, 'label': lbl, 'name': name}


class ToTensor(object):

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        img = image.transpose((2, 0, 1))
        img /= 256

        return {'image': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).long(),
                'name': name}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        h, w, _ = image.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = F.resize(image, (new_h, new_w))
        lbl = torch.unsqueeze(label, dim=0)
        lbl = F.resize(lbl, (new_h, new_w), interpolation=InterpolationMode.NEAREST)
        lbl = torch.squeeze(lbl)

        return {'image': img, 'label': lbl, 'name': name}


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        img = F.normalize(image, self.mean, self.std)

        return {'image': img, 'label': label, 'name': name}