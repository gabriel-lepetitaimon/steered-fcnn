import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, **hyperparameters):
        super(Model, self).__init__()
        self._hyperparameters = hyperparameters

    def __getattr__(self, item):
        if item == '_hyperparameters' or item not in self._hyperparameters:
            return super(Model, self).__getattr__(item)
        else:
            return self._hyperparameters[item]
