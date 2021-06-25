import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class adversarial():
    def __init__(self, option, dataset, model, epoch):
        self.dataset = dataset
        self.optimizer = optimizer
        self.model = model
        self.epoch = epoch

        self.loader = DataLoader(self.dataset, shuffle=False, batch_size=option.result['train']['batch_size'])
        self.perturbation = []

    def generate_permute_logit(self):
        self.criterion = None
        pass

    def generate_permute_feature(self):
        self.criterion = None
        pass

    def test(self):
        pass


