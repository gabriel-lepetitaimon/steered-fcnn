import torch
from torch import nn
import torch.nn.functional as F

from ..kbase_conv import KernelBase
from .steerable_filters import max_steerable_harmonics, radial_steerable_filter, cos_sin_ka
from ..utils import compute_padding, compute_conv_outputs_dim, clip_pad_center

from collections import OrderedDict
from typing import Union, Dict, List


class SteerableKernelBase(KernelBase):
    def __init__(self, base: 'list(np.array) [K,n,m]', n_kernel_by_k: 'dict {k -> n_k}'):
        """

        Args:
            base: kernels are assumed to be sorted by ascending k
            n_kernel_by_k:
        """
        super(SteerableKernelBase, self).__init__(base)

        # Sorting n_filter_by_k and removing invalid values
        self.n_filter_by_k = OrderedDict()
        for k in sorted(n_kernel_by_k.keys()):
            if n_kernel_by_k[k] > 0:
                self.n_filter_by_k[k] = n_kernel_by_k[k]
        n_kernel_by_k = self.n_filter_by_k

        # Statically store the maximum harmonic order (max_k) and all harmonics orders values (k_values)
        self.k_max = max(n_kernel_by_k.keys())
        self.k_values = list(sorted(n_kernel_by_k.keys()))

        # Store the number of kernel for k=0
        self._n_k0 = n_kernel_by_k.get(0, 0)

        # Store list and cumulative list of kernel count for each k
        self._n_kernel_by_k = []
        self._start_idx_by_k = [0]
        c = 0
        for k in range(0, self.k_max + 1):
            n_kernel = n_kernel_by_k.get(k, 0)
            self._n_kernel_by_k.append(n_kernel)
            c += n_kernel
            self._start_idx_by_k.append(c)

        self.K_equi = self._n_k0
        self.K_steer = self._start_idx_by_k[-1] - self.K_equi
        self.K = self.K_equi + 2*self.K_steer

        assert self.K == base.shape[0], 'The sum of n_kernel_by_k must be equal ' \
                                        'to the number of kernel in base (base.shape[0]).\n ' \
                                        f'(base.shape: {base.shape}, n_kernel_by_k sum: {self._start_idx_by_k[-1]})'

        self.kernels_info = []
        self.kernels_label = []

    def idx(self, k, real=True):
        return self.idx_real(k) if real else self.idx_imag(k)

    def idx_equi(self):
        return slice(None, self.K_equi)

    def idx_real(self, k=None):
        if k is None:
            return slice(self.K_equi, self.K_equi+self.K_steer)
        if k > self.k_max:
            return slice(self.K, self.K)    # Empty slice at the end of the list of kernels
        return slice(self._start_idx_by_k[k], self._start_idx_by_k[k+1])

    def idx_imag(self, k=None):
        if k is None:
            return slice(self.K_equi+self.K_steer, None)
        if k > self.k_max or k <= 0:
            return slice(self.K, self.K)    # Empty slice at the end of the list of kernels
        return slice(self.K_steer+self._start_idx_by_k[k], self.K_steer+self._start_idx_by_k[k+1])

    @property
    def base_equi(self):
        return self.base[self.idx_equi()]

    @property
    def base_real(self):
        return self.base[self.idx_real()]

    @property
    def base_y(self):
        return self.base[self.idx_imag()]

    def create_weights(self, n_in, n_out):
        from torch.nn.init import calculate_gain
        gain = calculate_gain('relu')
        w = torch.empty((n_out, n_in, self.K))
        std_equi = gain*torch.sqrt(3/(n_in*self.K_equi))     # Each kernel is assume to sum to 1

        nn.init.normal_(w[..., self.idx_equi()], std=std_equi)

        std_ortho = gain*torch.sqrt(3 / (n_in * self.K_steer))
        nn.init.normal_(w[..., self.idx_real()], std=std_ortho)
        nn.init.normal_(w[..., self.idx_imag()], std=std_ortho)

        return w

    @staticmethod
    def create_from_rk(kr: Union[int, Dict[int, List[int]]], std=.5, size=None, max_k=None):
        """


        Args:
            kr: A specification of which steerable filters should be included in the base.
                This parameter can be one of:
                    - a dictionary mapping harmonics order to a list of radius:
                      {k0: [r0, r1, ...], k1: [r2, r3, ...], ...}
                    - an integer interpreted as the wanted kernel size:
                      for every r <= kr/2, k will be set to be the maximum number of harmonics
                      before the apparition of aliasing artefact
            std: The standard deviation of the gaussian distribution which weights the kernels radially.
            size:
            max_k:

        Returns: A SteerableKernelBase parametrized by the corresponding kernels.

        """
        import numpy as np
        if isinstance(kr, int):
            rk = {r: np.arange(max_steerable_harmonics(r)+1) for r in range(kr)}
            kr = {}
            for r, K in rk.items():
                for k in K:
                    if max_k is not None and k > max_k:
                        break
                    if k in kr:
                        kr[k].append(r)
                    else:
                        kr[k] = [r]

        kernels_real, kernels_imag = [], []
        labels_real, labels_imag = [], []
        info_real, info_imag = [], []
        n_kernel_by_k = {}

        if size is None:
            r_max = max(max(R) for R in kr.values())
            size = int(np.ceil(2*(r_max+std)))
            size += int(1-(size % 2))    # Ensure size is odd

        for k in sorted(kr.keys()):
            R = kr[k]
            for r in sorted(R):
                if k in n_kernel_by_k:
                    n_kernel_by_k[k] += 1
                else:
                    n_kernel_by_k[k] = 1

                psi = radial_steerable_filter(size, k, r, std=std)
                # TODO: normalize?

                labels_real += [f'k{k}r{r}'+('r' if k > 0 else '')]
                info_real += [{'k': k, 'r': r, 'type': 'R'}]
                kernels_real += [psi.real]
                if k > 0:
                    labels_imag += [f'k{k}r{r}I']
                    info_real += [{'k': k, 'r': r, 'type': 'I'}]
                    kernels_imag += [psi.imag]

        B = SteerableKernelBase(np.stack(kernels_real + kernels_imag), n_kernel_by_k=n_kernel_by_k)
        B.kernels_label = labels_real + labels_imag
        B.kernels_info = info_real + info_imag
        return B

    def conv2d(self, input: torch.Tensor, weight: torch.Tensor, alpha: torch.Tensor = None,
               stride=1, padding='same', dilation=1) -> torch.Tensor:
        """
        Compute the convolution of `input` and this base's kernels steered by the angle `alpha`
        and premultiplied by `weight`.

        Args:
            input: Input tensor.
            weight: Weight for each couples of in and out features and for each kernels of this base.
            alpha: The angle by which the kernels are steered. (If None then alpha=0.)
                    To enhance the efficiency of the steering computation, α should not be provided as an angle but as a
                    vertical and horizontal projection: cos(α) and sin(α).
                    Hence this parameter can either be:
                        - A 5D tensor where  alpha[0]=cos(α) and alpha[1]=sin(α)
                        - A 6D tensor where  alpha[0,k-1]=cos(kα) and alpha[1,k-1]=sin(kα) for k=1:k_max
                    The last 4 dimensions allow to specify a different alpha for each output pixels [n, n_out, h, w].
                    (If any of these dimensions has a length of 1, alpha will be broadcasted along.)
                    Default: None
            stride: The stride of the convolving kernel. Can be a single number or a tuple (sH, sW).
                    Default: 1
            padding:  Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                      one of 'true', 'same' and 'full'.
                      Default: 'same'
            dilation: The spacing between kernel elements. Can be a single number or a tuple (dH, dW).
                      Default: 1

        Shape:
            input: (b, n_in, h, w)
            weight: (n_out, n_in, K)
            alpha: (2, [k_max], b, n_out, ~h, ~w)     (The last four dimensions are broadcastable
                                                       by replacing b, n_out, h or w by 1)
            return: (b, n_out, ~h, ~w)

        """
        if self.device != input.device:
            self.to(input.device)
        padding = compute_padding(padding, self.base.shape)
        conv_opts = dict(padding=padding, dilation=dilation, stride=stride)
        b, n_in, h, w = input.shape
        n_out, n_in_w, K = weight.shape
        assert n_in == n_in_w, 'Incoherent number of input neurons between the provided input and weight:\n' \
                               f'input.shape={input.shape} (n_in={n_in}), weight.shape={weight.shape} (n_in={n_in_w}).'
        h, w = compute_conv_outputs_dim(input.shape, weight.shape, **conv_opts)

        # If alpha=0 then the SteerableKernelBase behave like a simple KernelBase.
        if alpha is None:
            return super(SteerableKernelBase, self).composite_kernels_conv2d(input, weight, **conv_opts)

        # Otherwise if α != 0:
        # f = X⊛K_equi + Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]

        # computing f = X ⊛ K_equi ...
        if self._n_k0:
            f = F.conv2d(input, self.composite_equi_kernels(weight), **conv_opts)
        else:
            f = torch.zeros((b, n_out, h, w),
                            device=self.base.device, dtype=self.base.dtype)

        alpha = clip_pad_center(alpha, f.shape)

        # finally: f += Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]
        for k in self.k_values:
            if k == 0:
                continue

            if alpha.dim() == 5:
                if k == 0:
                    cos_sin_kalpha = alpha
                else:
                    cos_sin_kalpha = cos_sin_ka(alpha, cos_sin_kalpha)
            else:
                cos_sin_kalpha = alpha[:, k-1]

            f += cos_sin_kalpha[0] * F.conv2d(input, self.composite_steerable_kernels_real(weight, k=k), **conv_opts)
            f += cos_sin_kalpha[1] * F.conv2d(input, self.composite_steerable_kernels_imag(weight, k=k), **conv_opts)

        return f

    def composite_equi_kernels(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of all kernels for a the polar harmonic k=0,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]

        Shapes:
            weight: (n_out, n_in, K)
            return: (n_out, n_in, n, m)

        Returns: The composite kernel.
                 φ_ji0 = Σr[ ωR_ji0r ΨR_0r]
        """
        idx = self.idx_equi()
        return KernelBase.composite_kernels(weight[..., idx], self.base[idx])

    def composite_steerable_kernels_real(self, weight: torch.Tensor, k) -> torch.Tensor:
        """
        Compute φR_jik: the real part of the sum of all kernels for a specific polar harmonic k,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]
            k: The desired polar harmonic. (0 <= k <= self.max_k)

        Shapes:
            weight: (n_out, n_in, K)
            return: (n_out, n_in, n, m)

        Returns: The composite kernel.
                 φR_jik = Σr[ ωR_jikr ΨR_kr + ωI_jikr ΨI_kr]
        """
        if k == 0:
            return self.composite_equi_kernels(weight)

        real_idx = self.idx_real(k)
        imag_idx = self.idx_imag(k)
        w_real, w_imag = weight[..., real_idx], weight[..., imag_idx]
        psi_real, psi_imag = self.base[real_idx], self.base[imag_idx]

        return KernelBase.composite_kernels(w_real, psi_real) + KernelBase.composite_kernels(w_imag, psi_imag)

    def composite_steerable_kernels_imag(self, weight: torch.Tensor, k) -> torch.Tensor:
        """
        Compute φR_jik: the imaginary part of the sum of all kernels for a specific polar harmonic k,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]
            k: The desired polar harmonic. (0 <= k <= self.max_k)

        Shapes:
            weight: (n_out, n_in, K)
            return: (n_out, n_in, n, m)

        Returns: The composite kernel.
                 φI_jik = Σr[ ωR_jikr ΨI_kr - ωI_jikr ΨR_kr]
        """
        if k == 0:
            n_out, n_in, K = weight.shape
            K, n, m = self.base.shape
            return torch.zeros((n_out, n_in, n, m), device=self.base.device, dtype=self.base.dtype)
        real_idx = self.idx_real(k)
        imag_idx = self.idx_imag(k)
        w_real, w_imag = weight[..., real_idx], weight[..., imag_idx]
        psi_real, psi_imag = self.base[real_idx], self.base[imag_idx]

        return KernelBase.composite_kernels(w_real, psi_imag) - KernelBase.composite_kernels(w_imag, psi_real)

    def format_weights(self, weights, mean=True):
        from pandas import DataFrame
        import numpy as np
        from ..utils import iter_index
        data = dict(r=[_['r'] for _ in self.kernels_info],
                    k=[_['k'] for _ in self.kernels_info],
                    type=[_['type'] for _ in self.kernels_info])
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        s = weights.shape[:-1]
        if not len(s) or np.prod(s) == 1:
            data['weight'] = weights.flatten()
        else:
            for idx in iter_index(weights.shape):
                data[f'weights{list(idx)}'] = weights[idx]
            if mean:
                data['weights_mean'] = weights.mean(axis=tuple(range(len(s))))
                data['weights_std'] = weights.std(axis=tuple(range(len(s))))
        return DataFrame(data=data)

    def weights_dist(self, weights, Q=3):
        import numpy as np

        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights = weights.reshape((np.prod(weights.shape[:-1]), weights.shape[-1]))
        data = dict(r=[_['r'] for _ in self.kernels_info],
                    k=[_['k'] for _ in self.kernels_info],
                    type=[_['type'] for _ in self.kernels_info],
                    name=[f'r={_["r"]}, k={_["k"]}, {"Real" if _["type"]=="R" else "Imag"}' for _ in self.kernels_info])
        if isinstance(Q, int):
            Q = [i/(Q+1) for i in range(Q)]
        q = np.array(Q)*100
        q = q/2
        q = np.concatenate([50-q[::-1], [50], 50+q]).flatten()
        perc = np.percentile(weights, q, axis=0)
        data['median'] = perc[len(Q)]

        for i, q in enumerate(Q):
            data[f'q{i}'] = perc[-i-1]
            data[f'-q{i}'] = perc[i]
        return data

    def plot_weights(self, weights):
        pass