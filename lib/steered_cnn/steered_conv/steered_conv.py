import torch
from torch import nn
import math

from .steered_kbase import SteerableKernelBase
from .ortho_kbase import OrthoKernelBase
from ..utils.clip_pad import normalize_vector


DEFAULT_STEERABLE_BASE = SteerableKernelBase.from_steerable(4, max_k=5)
DEFAULT_ATTENTION_BASE = OrthoKernelBase.from_steerable(4)


class SteeredConv2d(nn.Module):
    def __init__(self, kernel, n_in, n_out=None, stride=1, padding='same', dilation=1, groups=1, bias=True,
                 steerable_base: SteerableKernelBase = DEFAULT_STEERABLE_BASE,
                 attention_base: SteerableKernelBase = None,
                 attention_mode='feature', normalize_steer_vec=None,
                 nonlinearity='relu', nonlinearity_param=None):
        """
        :param n_in:
        :param n_out:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param normalize_steer_vec: ignored if rho!=None when forward is called.
        """
        super(SteeredConv2d, self).__init__()

        if n_out is None:
            n_out = n_in
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if steerable_base is None:
            steerable_base = DEFAULT_STEERABLE_BASE
        elif isinstance(steerable_base, (int, dict)):
            steerable_base = SteerableKernelBase.from_steerable(steerable_base)
        self.steerable_base = steerable_base
        if attention_base is True:
            attention_base = DEFAULT_ATTENTION_BASE
        elif attention_base is False:
            attention_base = None
        elif isinstance(attention_base, (int, dict)):
            attention_base = SteerableKernelBase.from_steerable(attention_base)
        self.attention_base = attention_base

        # Weight
        self.weights = nn.Parameter(self.steerable_base.create_weights(n_in, n_out, nonlinearity, nonlinearity_param),
                                    requires_grad=True)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_out), requires_grad=True) if bias else None
            b = 1*math.sqrt(1/n_out)
            nn.init.uniform_(self.bias, -b, b)

        if self.attention_base is not None:
            self.attention_mode = attention_mode
            self.normalize_steer_vec = normalize_steer_vec
            n_att_out = n_out if attention_mode == 'feature' else 1
            self.attention_weigths = nn.Parameter(self.attention_base.create_weights(n_in, n_att_out,
                                                                                     nonlinearity='linear'),
                                                  requires_grad=True)

    def forward(self, x, alpha=None, rho=None):
        if alpha is None:
            alpha = self.attention_base.ortho_conv2d(x, self.attention_weigths,
                                                     stride=self.stride, padding=self.padding)
        if self.normalize_steer_vec and rho is None:
            alpha, rho = normalize_vector(alpha)
            if self.normalize_steer_vec == 'tanh':
                rho = torch.tanh(rho)
            elif self.normalize_steer_vec is True:
                rho = 1

        out = self.steerable_base.conv2d(x, self.weights, alpha=alpha, rho=rho,
                                         stride=self.stride, padding=self.padding, dilation=self.dilation)

        # Bias
        if self.bias is not None:
            out += self.bias[None, :, None, None]
        return out
