import torch
import torch.nn as nn
from .network import Incremental_Wrapper, Augment_Wrapper, Identity_Layer
from .loss import icarl_loss, rebalance_loss
from .layers import CosineLinear
from copy import deepcopy
import torch

def load_model(option, num_class, old_class, new_class, device, transform=None):
    if 'cifar' in option.result['data']['data_type']:
        from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    else:
        from .resnet_imagenet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

    # Load Network Architecture
    if option.result['network']['network_type'] == 'resnet18':
        model_enc = ResNet18()
    elif option.result['network']['network_type'] == 'resnet34':
        model_enc = ResNet34()
    else:
        raise('Select Proper Network Type')

    # Training Type
    if option.result['train']['train_type'] == 'icarl':
        model_fc = nn.Linear(model_enc.num_feature, num_class, bias=True)

        if option.result['exemplar']['augment'] is None:
            model = Incremental_Wrapper(option, model_enc=model_enc, model_fc=model_fc, old_class=old_class, new_class=new_class, device=device)
        else:
            model = Augment_Wrapper(option, model_enc, model_fc, old_class, new_class, device, transform)

    elif option.result['train']['train_type'] == 'naive':
        pass

    elif option.result['train']['train_type'] == 'rebalance':
        model_fc = CosineLinear(model_enc.num_feature, num_class)

        if option.result['exemplar']['augment'] is None:
            model = Incremental_Wrapper(option, model_enc=model_enc, model_fc=model_fc, old_class=old_class, new_class=new_class, device=device)
        else:
            model = Augment_Wrapper(option, model_enc, model_fc, old_class, new_class, device, transform)

    else:
        raise('Select Proper training type')

    return model


def load_optimizer(option, params):
    optimizer = option.result['optim']['optimizer']
    lr = option.result['optim']['lr']
    weight_decay = option.result['optim']['weight_decay']

    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, nesterov=option.result['optim']['nesterov'], momentum=option.result['optim']['momentum'])
    else:
        raise('Selec proper optimizer')


def load_scheduler(option, optimizer):
    if option.result['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
    elif option.result['train']['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 63, 80], gamma=0.2)
    elif option.result['train']['scheduler'] is None:
        scheduler = None
    else:
        raise('select proper scheduler')

    return scheduler

def load_loss(option, old_class, new_class):
    train_type = option.result['train']['train_type']
    if train_type == 'naive':
        criterion = nn.CrossEntropyLoss()
    elif train_type == 'icarl':
        criterion = icarl_loss(old_class, new_class)
    elif train_type == 'rebalance':
        criterion = nn.CrossEntropyLoss()
    else:
        raise('select proper train_type')

    return criterion