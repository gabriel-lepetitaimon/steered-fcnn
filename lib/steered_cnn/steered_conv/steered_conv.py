import torch
from torch import nn
import math

from .steered_kbase import SteerableKernelBase
from .ortho_kbase import OrthoKernelBase
from ..utils.clip_pad import normalize_vector

DEFAULT_STEERABLE_BASE = SteerableKernelBase.create_radial(3)
DEFAULT_ATTENTION_BASE = OrthoKernelBase.create_radial(3)


class SteeredConv2d(nn.Module):
    def __init__(self, n_in, n_out=None, stride=1, padding='same', dilation=1, groups=1, bias=True,
                 steerable_base: SteerableKernelBase = None,
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
        self.steerable_base = SteerableKernelBase.parse(steerable_base, DEFAULT_STEERABLE_BASE)
        self.attention_base = OrthoKernelBase.parse(attention_base, default=DEFAULT_ATTENTION_BASE)

        # Weight
        self.weights = nn.Parameter(self.steerable_base.init_weights(n_in, n_out, nonlinearity, nonlinearity_param),
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
            self.attention_weigths = nn.Parameter(self.attention_base.init_weights(n_in, n_att_out,
                                                                                   nonlinearity='linear'),
                                                  requires_grad=True)

    def forward(self, x, alpha=None, rho=None):
        if alpha is None:
            if self.attention_base:
                alpha = self.attention_base.ortho_conv2d(x, self.attention_weigths,
                                                         stride=self.stride, padding=self.padding)
            else:
                raise ValueError('Either attention_base or alpha should be provided when computing the result of a '
                                 'SteeredConv2d module.')
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

    
class SteeredConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out=None, stride=2, padding='same', output_padding=0,
                 dilation=1, groups=1, bias=True,
                 steerable_base: SteerableKernelBase = None,
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
        super(SteeredConvTranspose2d, self).__init__()

        if n_out is None:
            n_out = n_in
        
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        if steerable_base is None:
            steerable_base = SteerableKernelBase.create_radial(max(stride) if isinstance(stride, tuple) else stride)
        self.steerable_base = steerable_base
        if attention_base is True:
            attention_base = DEFAULT_ATTENTION_BASE
        elif attention_base is False:
            attention_base = None
        elif isinstance(attention_base, int):
            attention_base = SteerableKernelBase.create_radial(attention_base)
        self.attention_base = attention_base

        # Weight
        self.weights = nn.Parameter(self.steerable_base.init_weights(n_in, n_out, nonlinearity, nonlinearity_param),
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
            self.attention_weigths = nn.Parameter(self.attention_base.init_weights(n_in, n_att_out,
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

        out = self.steerable_base.conv_transpose2d(x, self.weights, alpha=alpha, rho=rho, stride=self.stride,
                                                   padding=self.padding, output_padding=self.output_padding,
                                                   dilation=self.dilation)

        # Bias
        if self.bias is not None:
            out += self.bias[None, :, None, None]
        return out