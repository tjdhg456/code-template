import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

def load_model(option):
    model = resnet18(pretrained=False)
    return model

def load_optimizer(option, param):
    optim = torch.optim.SGD(param, lr=1e-3)
    return optim

def load_scheduler(option):
    return None

def load_loss(option):
    criterion = nn.CrossEntropyLoss()
    return criterion