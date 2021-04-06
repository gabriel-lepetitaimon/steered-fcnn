import torch
from torch import nn

from .steered_kbase import SteerableKernelBase


_DEFAULT_STEERABLE_BASE = SteerableKernelBase.create_from_rk(4, max_k=5)


class SteeredConv2d(nn.Module):
    def __init__(self, n_in, n_out=None, steerable_base: SteerableKernelBase = None,
                 stride=1, padding='auto', dilation=1, groups=1, bias=True):
        """
        :param n_in:
        :param n_out:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """
        super(SteeredConv2d, self).__init__()
        
        if n_out is None:
            n_out = n_in
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if steerable_base is None:
            steerable_base = _DEFAULT_STEERABLE_BASE
        elif isinstance(steerable_base, (int, dict)):
            steerable_base = SteerableKernelBase.create_from_rk(steerable_base)
        self.base = steerable_base

        # Weight
        self.weights = nn.Parameter(self.base.create_weights(n_in, n_out), requires_grad=True)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_out), requires_grad=True) if bias else None
            b = 1*torch.sqrt(3/n_out)
            nn.init.uniform_(self.bias, -b, b)

    def forward(self, x, alpha=None):
        out = self.base.conv2d(x, self.weights, alpha=alpha,
                               stride=self.stride, padding=self.padding, dilation=self.dilation)

        # Bias
        if self.bias is not None:
            out += self.bias[None, :, None, None]
        return out
