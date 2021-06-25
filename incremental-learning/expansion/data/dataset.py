from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from random import shuffle
import torch

def load_cifar10(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    ])


    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    ])

    tr_dataset = torchvision.datasets.CIFAR10(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type']), train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type']), train=False, download=True, transform=val_transform)
    ex_dataset = torchvision.datasets.CIFAR10(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type']), train=True, download=True, transform=val_transform)
    return tr_dataset, val_dataset, ex_dataset


def load_cifar100(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    ])


    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    ])

    tr_dataset = torchvision.datasets.CIFAR100(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type']), train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR100(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type']), train=False, download=True, transform=val_transform)
    ex_dataset = torchvision.datasets.CIFAR100(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type']), train=True, download=True, transform=val_transform)
    return tr_dataset, val_dataset, ex_dataset


def load_imagenet(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'], 'val'), transform=val_transform)
    ex_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'], 'train'), transform=val_transform)
    return tr_dataset, val_dataset, ex_dataset


def load_food101(option):
    tr_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tr_dataset = torchvision.datasets.ImageFolder(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'], 'val'), transform=val_transform)
    ex_dataset = torchvision.datasets.ImageFolder(root=os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'], 'train'), transform=val_transform)
    return tr_dataset , val_dataset, ex_dataset


def load_data(option, data_type='train'):
    if option.result['data']['data_type'] == 'cifar10':
        tr_d, val_d, ex_d = load_cifar10(option)
    elif option.result['data']['data_type'] == 'cifar100':
        tr_d, val_d, ex_d = load_cifar100(option)
    elif option.result['data']['data_type'] == 'imagenet':
        tr_d, val_d, ex_d = load_imagenet(option)
    elif option.result['data']['data_type'] == 'food101':
        tr_d, val_d, ex_d = load_food101(option)
    else:
        raise('select appropriate dataset')

    if data_type == 'train':
        return tr_d
    elif data_type == 'val':
        return val_d
    else:
        return ex_d


class transform_module():
    def __init__(self, option):
        if 'cifar' in option.result['data']['data_type']:
            mu = np.array([0.5071,  0.4866,  0.4409])
            std = np.array([0.2009,  0.1984,  0.2023])
        elif option.result['data']['data_type'] == 'imagenet':
            mu = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        elif option.result['data']['data_type'] == 'food101':
            mu = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        else:
            raise ('select appropriate dataset')

        self.mu = mu
        self.std = std

    def normalize(self, img):
        img = torch.squeeze(img)
        img = transforms.Normalize(self.mu, self.std)(img)
        img = torch.unsqueeze(img, dim=0)
        return img

    def un_normalize(self, img):
        transform_un_normalize = transforms.Compose([transforms.Normalize((0,0,0), 1/self.std),
                                                     transforms.Normalize(-self.mu, (1,1,1))])
        img = torch.squeeze(img)
        img = transform_un_normalize(img)
        img = torch.unsqueeze(img, dim=0)
        return img


class IncrementalSet(Dataset):
    def __init__(self, dataset, dataset_exemplar, start, target_list, shuffle_label=False):
        self.dataset = dataset
        self.dataset_label = np.array(self.dataset.targets)
        self.target_index = np.concatenate([np.where(self.dataset_label == ix)[0] for ix in target_list], axis=0)
        self.index_list = list(range(len(self.target_index)))

        self.dataset_exemplar = dataset_exemplar

        self.start = start
        self.shuffle = shuffle_label

        self.exemplary = []

        if self.shuffle:
            shuffle(self.index_list)

    def update_exemplar(self, exemplar):
        self.exemplary = [e for ex in exemplar for e in ex]
        self.index_list = list(range(len(self.target_index) + len(self.exemplary)))
        shuffle(self.index_list)

    def get_image_class(self, label):
        self.target_label_index = np.where(self.dataset_label == label)[0]
        return [self.dataset_exemplar.__getitem__(index) for index in self.target_label_index]

    def __len__(self):
        return len(self.target_index) + len(self.exemplary)

    def __getitem__(self, index):
        index = self.index_list[index]

        if index < len(self.target_index):
            image, label = self.dataset.__getitem__(self.target_index[index])
        else:
            image, label = self.exemplary[index - len(self.target_index)]

        return image, label