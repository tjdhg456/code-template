import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from .network import Incremental_Wrapper

def load_model(option, num_class):
    model_cls = resnet18(pretrained=False, num_classes=num_class)
    model = Incremental_Wrapper(option, model_cls)
    return model

def load_optimizer(option, param):
    optim = torch.optim.SGD(param, lr=option.result['train']['lr'], momentum=0.9, weight_decay=5e-4)
    return optim

def load_scheduler(option, optimizer):
    if option.result['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
    elif option.result['train']['scheduler'] is None:
        scheduler = None
    else:
        raise('select proper scheduler')

    return scheduler

def load_loss(option):
    criterion = nn.CrossEntropyLoss()
    return criterion