import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, hyperparameters=None):
        super(Model, self).__init__()
        if hyperparameters is None:
            hyperparameters = {}

        self.hyperparameters = hyperparameters
