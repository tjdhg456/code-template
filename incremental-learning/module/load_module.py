import torch
import torch.nn as nn
from .network import Incremental_Wrapper, Icarl_Wrapper, Identity_Layer
from .loss import icarl_loss
from .resnet import resnet18_cbam, resnet34_cbam, resnet50_cbam, resnet152_cbam
from copy import deepcopy
import torch


def load_model(option, num_class):
    model_enc = resnet18_cbam(pretrained=False)
    model_fc = nn.Linear(model_enc.num_feature, num_class, bias=True)

    if option.result['train']['train_type'] == 'icarl':
        model = Icarl_Wrapper(option, model_enc=model_enc, model_fc=model_fc)

    else:
        model = Incremental_Wrapper(option, model_enc=model_enc, model_fc=model_fc)
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
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum = option.result['optim']['momentum'])
    else:
        raise('Selec proper optimizer')


def load_scheduler(option, optimizer):
    if option.result['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
    elif option.result['train']['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 62, 80], gamma=0.2)
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
    else:
        raise('select proper train_type')

    return criterion