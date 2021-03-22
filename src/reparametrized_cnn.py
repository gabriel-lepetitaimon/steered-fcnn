import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .torch_utils import *


class KernelBase:
    def __init__(self, base: 'list(np.array) [n][f,h,w]'):
        self.base = [torch.from_numpy(_).to(dtype=torch.float) for _ in base]

    @staticmethod
    def cardinal_base(size=3):
        base = []
        for k in range(1, size+1, 2):
            if k==1:
                base.append(np.ones((1,1,1)))
            else:
                kernels = []
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[0, i] = 1
                    kernels.append(K)
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[i, -1] = 1
                    kernels.append(K)
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[-1, -1-i] = 1
                    kernels.append(K)
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[-1-i, 0] = 1
                    kernels.append(K)
                base.append(np.stack(kernels))
        return KernelBase(base)

    def create_params(self, n_in, n_out):
        R = len(self.base)
        K = sum([self.base[r].shape[0] for r in range(R)])
        w = nn.Parameter(torch.empty((n_out, n_in, K)), requires_grad=True)
        b = np.sqrt(3/(n_in*K))
        nn.init.uniform_(w, -b, b)
        return w

    @property
    def device(self):
        return self.base[0].get_device()

    def to(self, *args, **kwargs):
        for b in self.base:
            b.to(*args, **kwargs)
        return self

    # --- Composite Kernels ---
    @staticmethod
    def composite_kernels(base_kernels, weight: 'torch.Tensor [n_out, n_in, n_k]'):
        kernels = None
        k0 = 0
        W = weight
        n_out, n_in, n_k = W.shape
        for K in base_kernels:
            k, h_k, w_k = K.shape
            k1 = k0+k
            K = K.reshape(k, h_k*w_k)
            # W: [n_out,n_in,k0:k0+k] * F: [f,hw] -> [n_out, n_in, hw]
            kernel = torch.matmul(W[:, :, k0:k1], K).reshape(n_out, n_in, h_k, w_k)
            if kernels is None:
                kernels = kernel
            else:
                kernels = sum(pad_tensors(kernels, kernel))
            k0 = k1
        return kernels

    def composite_kernels_conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [n_out, n_in, k]',
                                 stride=1, padding='auto', dilation=1, groups=1):
        W = KernelBase.composite_kernels(self.base, weight)
        padding = get_padding(padding, W.shape)
        return F.conv2d(input, W, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # --- Preconvolve Base ---
    @staticmethod
    def preconvolve_base(input: 'torch.Tensor [b,i,w,h]', base_kernels,
                         stride=1, padding='auto', dilation=1):
        base = None
        b, n_in, h, w = input.shape
        input = input.reshape(b * n_in, 1, h, w)

        for kernel in base_kernels:
            n_k, h_k, w_k = kernel.shape
            pad = get_padding(padding, (h_k, w_k))

            K = F.conv2d(input, kernel[:, None, :, :], stride=stride, padding=pad, dilation=dilation)
            h, w = K.shape[-2:]     # h and w after padding, K.shape: [b*n_in, n_k, ~h, ~w]
            K = K.reshape(b, n_in, n_k, h, w)

            if base is None:
                base = K
            else:
                base = torch.cat(clip_tensors(base, K), dim=2)
        return base

    def preconvolved_base_conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [n_out, n_in, k]',
                                 stride=1, padding='auto', dilation=1):
        bases = KernelBase.preconvolve_base(input, self.base, stride=stride, padding=padding, dilation=dilation)
        b, n_in, k, h, w = bases.shape
        n_out, n_in_w, k_w = weight.shape

        assert n_in == n_in_w, f"The provided inputs and weights have different number of input neuron:\\ " +\
                               f"x.shape[1]=={n_in}, weigth.shape[1]=={n_in_w}."
        assert k == k_w, f"The provided weights have an incorrect number of kernels:\\ " +\
                         f"weight.shape[2]=={k_w}, but should be {k}."

        K = bases.permute(0, 3, 4, 1, 2).reshape(b, h, w, n_in*k)
        W = weight.reshape(n_out, n_in*k).transpose(0, 1)
        f = torch.matmul(K, W)  # K:[b,h,w,n_in*k] x W:[n_in*k, n_out] -> [b,h,w,n_out]
        return f.permute(0, 3, 1, 2)    # [b,n_out,h,w]


