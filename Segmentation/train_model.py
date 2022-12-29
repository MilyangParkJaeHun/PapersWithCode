from __future__ import print_function, division
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from data_utils import CustomDataset
from transformation import *
from SemanticSegmentation.FCN.FCN import VGGNet, FCN8s

color_map = (
    (63, 63, 63),
    (0, 102, 0),
    (204, 255, 204),
    (255, 102, 102),
    (0, 255, 255),
    (204, 204, 51),
    (51, 51, 204),
    (0, 255, 64),
    (0, 102, 153),
    (153, 0, 102),
    (204, 51, 204),
    (0, 165, 255),
    (3, 107, 252),
    (128, 128, 0))

saved_path = '/home/park/workspace/PapersWithCode/Segmentation/validation'


def convert_index_to_bgr(index_mat):
    bgr_image = cv2.merge((index_mat, index_mat, index_mat))

    for index, color in enumerate(color_map):
        bgr_image[index_mat == index] = color
    bgr_image[index_mat > index] = (0, 0, 0)

    return bgr_image

def save_prediction(name, image, label, predict):
    image = image.squeeze(0).cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = 256 * ((image * 0.5) + 0.5)
    image = image.astype(np.uint8)

    label = label.squeeze(0).cpu().numpy()
    label = convert_index_to_bgr(label)

    predict = predict.squeeze(0).cpu().numpy()
    predict = convert_index_to_bgr(predict)

    prediction_image = np.concatenate([image, label, predict], 1)

    cv2.imwrite(os.path.join(saved_path, name), prediction_image)


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.localtime(time.time())
        time_stamp = '_'.join([
            str(start_time.tm_year),
            str(start_time.tm_mon),
            str(start_time.tm_mday),
            str(start_time.tm_hour),
            str(start_time.tm_min),
            str(start_time.tm_sec)])
        print(f'Epoch {epoch}/{num_epochs - 1}  {time_stamp}')
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in data_loader[phase]:
                inputs = data['image']
                labels = data['label']
                names = data['name']
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)['out']
                    _, predicts = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'valid':
                        for name, image, label, predict in zip(names, inputs, labels, predicts):
                            save_prediction(name, image, label, predict)

                batch_loss = loss.item() * inputs.size(0)
                running_loss += batch_loss
                running_corrects += torch.sum(predicts == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    data_params = dict()
    data_params['root_dir'] = '/home/park/DATA/2022_11_23/trails'
    data_params['image_size'] = (128, 256)
    data_params['num_class'] = 14

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            ConvertLabel(),
            ToTensor(),
            Rescale(data_params['image_size']),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            ConvertLabel(),
            ToTensor(),
            Rescale(data_params['image_size']),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    dataset = {mode:
        CustomDataset(params=data_params, mode=mode, transform=data_transforms[mode])
        for mode in ['train', 'valid']}
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'valid']}

    data_loader = {}
    data_loader['train'] = DataLoader(dataset['train'],
                            batch_size=16,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            # pin_memory_device=device,
                            prefetch_factor=2)

    data_loader['valid'] = DataLoader(dataset['valid'],
                            batch_size=16,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            # pin_memory_device=device,
                            prefetch_factor=2)

    # model = models.segmentation.fcn_resnet50(num_classes=14).to(device)
    vgg16 = VGGNet('vgg16', 3).to(device)
    model = FCN8s(vgg16, data_params['num_class']).to(device)

    # model = VGGNet16(3).to(device)

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train(model, data_loader, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
