import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, **hyperparameters):
        super(Model, self).__init__()
        self.hyperparameters = hyperparameters

    def __getattribute__(self, item):
        if item in self.hyperparameters:
            return self.hyperparameters[item]
