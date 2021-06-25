import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
import math

class CosineLinear(Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.requires_grad = True

        self.bias = None

        self.sigma = Parameter(torch.Tensor(1))
        self.sigma.requires_grad = True

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        out = self.sigma * out
        return out

