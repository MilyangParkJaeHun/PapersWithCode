from __future__ import print_function, division
import cv2
import os
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformation import *


class CustomDataset(Dataset):

    def __init__(self, params,  mode='train', transform=None):
        self.mode = mode
        self.root_dir = params['root_dir']
        self.image_size = params['image_size']
        self.num_class = params['num_class']

        self.transform = transform
        self.image = glob.glob(os.path.join(self.root_dir, self.mode, 'img', '*.png'))
        self.label = glob.glob(os.path.join(self.root_dir, self.mode, 'lbl', '*.png'))

        assert len(self.image) == len(self.label)

        self.class_weight = self.__calculate_class_weight(self.label)

        print('Loaded {}, total data size : {}'.format(
            os.path.join(self.root_dir, mode),
            len(self.image)))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.int()

        image_name = self.image[idx]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR).astype(np.float32)

        label_path = self.label[idx]
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)

        sample = {'image': image.copy(order='C'),
                  'label': label.copy(order='C'),
                  'name': os.path.basename(image_name)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __calculate_class_weight(self, label_list):
        class_count = np.zeros(self.num_class)

        for label_path in label_list:
            label = cv2.imread(label_path, cv2.IMREAD_COLOR)
            label = cv2.resize(
                label,
                self.image_size,
                interpolation=cv2.INTER_NEAREST)
            label = cv2.split(label)[0]

            histogram = cv2.calcHist(label, [0], None, [self.num_class], [0, self.num_class])
            histogram = np.array(histogram).squeeze()

            class_count += histogram

        class_percent = class_count / np.sum(class_count)

        e = 1.02
        class_weight = 1 / np.log(class_percent + e)

        return class_weight


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            ConvertLabel(),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            ConvertLabel(),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    data_params = dict()
    data_params['root_dir'] = ''
    data_params['image_size'] = ()
    data_params['num_class'] = 0

    dataset = {mode:
        CustomDataset(params=data_params, mode=mode, transform=data_transforms[mode])
        for mode in ['train', 'valid']}

    data_loader = {}
    data_loader['train_dataset'] = DataLoader(dataset['train'],
                            batch_size=64,
                            shuffle=True,
                            num_workers=8)

    data_loader['valid_dataset'] = DataLoader(dataset['valid'],
                            batch_size=64,
                            shuffle=False,
                            num_workers=8)

    for i_batch, sample_batched in enumerate(data_loader['train_dataset']):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['label'].size())

        images, labels, names = sample_batched['image'], sample_batched['label'], sample_batched['name']
        image = (images[0].numpy())
        label = labels[0].numpy()
        name = names[0]

        image = image.transpose(1, 2, 0)
        image = 256*((image * 0.5) + 0.5)

        image = image.astype(np.uint8)
        cv2.imshow(name, image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break