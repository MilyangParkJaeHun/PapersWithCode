import cv2
import torch
import numpy as np
from skimage import io, transform
from torchvision.transforms import functional as F
from torchvision import transforms


class ConvertLabel(object):

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        lbl = cv2.split(label)[0]

        return {'image': image, 'label': lbl, 'name': name}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        h, w, c = image.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w), cv2.INTER_LINEAR)
        lbl = cv2.resize(label, (new_h, new_w), cv2.INTER_NEAREST)

        return {'image': img, 'label': lbl, 'name': name}


class ToTensor(object):

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        img = image.transpose((2, 0, 1))
        img /= 256

        return {'image': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).long(),
                'name': name}


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        img = F.normalize(image, self.mean, self.std)

        return {'image': img, 'label': label, 'name': name}