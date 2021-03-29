from torch import nn
import numpy as np
from .torch_utils import *
from .utils import warn
from .reparametrized_cnn import KernelBase
from collections import OrderedDict


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
        self.max_k = max(n_kernel_by_k.keys())
        self.k_values = list(sorted(n_kernel_by_k.keys()))

        # Store the number of kernel for k=0
        self._n_k0 = n_kernel_by_k.get(0, 0)

        # Store list and cumulative list of kernel count for each k
        self._n_kernel_by_k = []
        self._start_idx_by_k = [0]
        c = 0
        for k in range(0, self.max_k+1):
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
        if k > self.max_k:
            return slice(self.K, self.K)    # Empty slice at the end of the list of kernels
        return slice(self._start_idx_by_k[k], self._start_idx_by_k[k+1])

    def idx_imag(self, k=None):
        if k is None:
            return slice(self.K_equi+self.K_steer, None)
        if k > self.max_k or k <= 0:
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
        std_equi = gain*np.sqrt(3/(n_in*self.K_equi))     # Each kernel is assume to sum to 1

        nn.init.normal_(w[..., self.idx_equi()], std=std_equi)

        std_ortho = gain*np.sqrt(3 / (n_in * self.K_steer * 2))
        nn.init.normal_(w[..., self.idx_real()], std=std_ortho)
        nn.init.normal_(w[..., self.idx_imag()], std=std_ortho)

        return w

    @staticmethod
    def create_from_rk(kr, std=.5, size=None, max_k=None):
        """
        kr: A specification of which steerable filters should be included in the base.
        This parameter can one of:
            - a dictionary mapping harmonics order to a list of radius:
                {k0: [r0, r1, ...], k1: [r2, r3, ...], ...}
            - an integer interpreted as the wanted kernel size:
                for every r <= kr/2, k will be set to be the maximum number of harmonics before the apparition of aliasing artefact
        std: The standard deviation of the gaussian distribution which weights the kernels radially.
        return: A SteerableKernelBase parametrized by the corresponding kernels.
        """
        if isinstance(kr, int):
            from .rot_utils import max_steerable_harmonics
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

        from .rot_utils import radial_steerable_filter
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
                info_real += [{'k':k, 'r': r, 'type':'R'}]
                kernels_real += [psi.real]
                if k > 0:
                    labels_imag += [f'k{k}r{r}I']
                    info_real += [{'k':k, 'r': r, 'type':'I'}]
                    kernels_imag += [psi.imag]

        B = SteerableKernelBase(np.stack(kernels_real + kernels_imag), n_kernel_by_k=n_kernel_by_k)
        B.kernels_label = labels_real + labels_imag
        B.kernels_info = info_real + info_imag
        return B

    def conv2d(self, input: 'torch.Tensor [b,n_in,h,w]', weight: 'np.array [n_out,n_in,K]', alpha: 'torch.Tensor [b,?n_out,h,w]' = None,
               stride=1, padding='auto', dilation=1) -> 'torch.Tensor [b,n_out,~h,~w]':
        if self.device != input.device:
            self.to(input.device)
        padding = get_padding(padding, self.base.shape)
        conv_opts = dict(padding=padding, dilation=dilation, stride=stride)
        b, n_in, h, w = input.shape
        n_out, n_in_w, K = weight.shape
        assert n_in == n_in_w, 'Incoherent number of input neurons between the provided input and weight:\n' \
                              f'input.shape={input.shape} (n_in={n_in}), weight.shape={weight.shape} (n_in={n_in_w}).'

        # If alpha=0 then the SteerableKernelBase behave like a simple KernelBase.
        if alpha is None:
            return super(SteerableKernelBase, self).composite_kernels_conv2d(input, weight, **conv_opts)

        # Otherwise if α != 0:
        # f = X⊛K_equi + Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]

        # computing f = X ⊛ K_equi ...
        if self._n_k0:
            f = F.conv2d(input, self.composite_equi_kernels(weight), **conv_opts)
        else:
            f = torch.zeros((b, n_out)+get_outputs_dim(input.shape, weight.shape, **conv_opts),
                            device=self.base.device, dtype=self.base.dtype)

        # then, preparing α...
        if isinstance(alpha, (float, int)) or (isinstance(alpha, torch.Tensor) and alpha.dim == 0):
            alpha = alpha * torch.ones((1, 1, 1, 1), dtype=self.base.dtype, device=self.base.device)
        elif isinstance(alpha, torch.Tensor):
            if alpha.dim() == 3:
                alpha = alpha[:, None, :, :]
            alpha = clip_pad_center(alpha, f.shape)

        # finally: f += Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]
        for k in self.k_values:
            if k == 0:
                continue
            if isinstance(alpha, tuple):
                cos_kalpha = clip_pad_center(alpha[0][k-1], f.shape)
                sin_kalpha = clip_pad_center(alpha[1][k-1], f.shape)
            else:
                cos_kalpha = torch.cos(k*alpha)
                sin_kalpha = torch.sin(k*alpha)
            f += cos_kalpha * F.conv2d(input, self.composite_steerable_kernels_real(weight, k=k), **conv_opts)
            f += sin_kalpha * F.conv2d(input, self.composite_steerable_kernels_imag(weight, k=k), **conv_opts)

        return f

    def composite_equi_kernels(self, weight: 'torch.Tensor [n_out, n_in, K]') -> '[n_out, n_in, n, m]':
        """
        Compute the sum of all kernels for a the polar harmonic k=0,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base (shape: [n_out, n_in, K])
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]

        Returns: The composite kernel. (shape: [n_out, n_in, n, m])
                 φ_ji0 = Σr[ ωR_ji0r ΨR_0r]
        """
        idx = self.idx_equi()
        return KernelBase.composite_kernels(weight[..., idx], self.base[idx])

    def composite_steerable_kernels_real(self, weight: 'torch.Tensor [n_out, n_in, K]', k) -> '[n_out, n_in, n, m]':
        """
        Compute φR_jik: the real part of the sum of all kernels for a specific polar harmonic k,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base (shape: [n_out, n_in, K])
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]
            k: The desired polar harmonic. (0 <= k <= self.max_k)

        Returns: The composite kernel. (shape: [n_out, n_in, n, m])
                 φR_jik = Σr[ ωR_jikr ΨR_kr + ωI_jikr ΨI_kr]
        """
        if k == 0:
            return self.composite_equi_kernels(weight)

        real_idx = self.idx_real(k)
        imag_idx = self.idx_imag(k)
        w_real, w_imag = weight[..., real_idx], weight[..., imag_idx]
        psi_real, psi_imag = self.base[real_idx], self.base[imag_idx]

        return KernelBase.composite_kernels(w_real, psi_real) + KernelBase.composite_kernels(w_imag, psi_imag)

    def composite_steerable_kernels_imag(self, weight: 'torch.Tensor [n_out, n_in, K]', k) -> '[n_out, n_in, n, m]':
        """
        Compute φR_jik: the imaginary part of the sum of all kernels for a specific polar harmonic k,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base (shape: [n_out, n_in, K])
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]
            k: The desired polar harmonic. (0 <= k <= self.max_k)

        Returns: The composite kernel. (shape: [n_out, n_in, n, m])
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
            b = 1*np.sqrt(3/(n_out))
            nn.init.uniform_(self.bias, -b, b)

    def forward(self, x, alpha=None):
        out = self.base.conv2d(x, self.weights, alpha=alpha,
                               stride=self.stride, padding=self.padding, dilation=self.dilation)

        # Bias
        if self.bias is not None:
            out += self.bias[None, :, None, None]
        return out


class SteeredConvBN(nn.Module):
    def __init__(self, n_in, n_out=None, steerable_base=None,
                 stride=1, padding='auto', dilation=1, groups=1, bn=False, relu=True):
        super(SteeredConvBN, self).__init__()
        
        self._bn = bn
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        self.conv = SteeredConv2d(n_in, n_out, steerable_base=steerable_base, stride=stride, groups=groups,
                                  padding=padding, bias=not bn, dilation=dilation)
        bn_relu = []
        if bn:
            bn_relu += [nn.BatchNorm2d(self.n_out)]
            if relu:
                bn_relu += [nn.ReLU()]
        elif relu:
            bn_relu += [nn.SELU()]

        self.bn_relu = nn.Sequential(*bn_relu)
        
    def forward(self, x, alpha=None):
        x = self.conv(x, alpha=alpha)
        return self.bn_relu(x)
    
    @property
    def bn(self):
        if self._bn:
            return self.model[0]
        return None

    @property
    def relu(self):
        return self.model[1 if self._bn else 0]


_gprof = torch.Tensor([1, 2, 1])
_gkernel = torch.Tensor([-1, 0, 1])[:, np.newaxis] * _gprof[np.newaxis, :]
_gkernel_t = _gkernel.transpose(0,1).unsqueeze(0).unsqueeze(0)
_gkernel = _gkernel.unsqueeze(0).unsqueeze(0)


def smooth(t, smooth_std, device=None):
    if isinstance(smooth_std, (int,float)):
        from .gaussian import get_gaussian_kernel2d
        smooth_ksize = int(np.ceil(3*smooth_std))
        smooth_ksize += 1-(smooth_ksize % 2)
        smooth_kernel = get_gaussian_kernel2d((smooth_ksize, smooth_ksize), (smooth_std, smooth_std)) \
            .unsqueeze(0).unsqueeze(0)
        if device is not None:
            smooth_kernel = smooth_kernel.to(device)

    elif isinstance(smooth_std, torch.Tensor):
        smooth_kernel = smooth_std.device(device) if device is not None else smooth_std
        smooth_kernel.unsqueeze(0).unsqueeze(0)
        smooth_ksize = smooth_kernel.shape[-1]
    else:
        raise TypeError
    return F.conv2d(t, smooth_kernel, padding=(smooth_ksize-1)//2)


def grad(t, smooth_std=1.2, device=None):
    with torch.no_grad():
        b, c, h, w = t.shape
        t = t.reshape(b*c,1,h,w)

        if smooth_std:
            t = smooth(t, smooth_std, device=device)


        gkernel_t = _gkernel_t.device(device) if device else _gkernel_t
        gkernel = _gkernel_t.device(device) if device else _gkernel

        gx = F.conv2d( t, gkernel_t, padding=1)
        gy = F.conv2d( t, gkernel, padding=1)

        return gy.reshape(b,c,h,w), gx.reshape(b,c,h,w)


def hessian_principal_direction(t, smooth_std=1.2, max_dir=True, device=None, logs=None):
    """ Compute the principal direction (first eigen vector) of the hessian for each pixels in t.
    t.shape = (b, c, h, w)
    Return vy, vx, vr (namely horizontal and vertical component of the unitary vector of the principal hessian direction, and norm of the vector before normalization).
    """
    #global _smooth_kernel
    with torch.no_grad():
        b, c, h, w = t.shape
        t = t.reshape(b*c,1,h,w)

        if smooth_std:
            t = smooth(t, smooth_std, device=device)

        gkernel_t = _gkernel_t.to(device) if device else _gkernel_t
        gkernel = _gkernel_t.to(device) if device else _gkernel

        gx = F.conv2d( t, gkernel_t, padding=1)
        gy = F.conv2d( t, gkernel, padding=1)
        gxx = F.conv2d(gx, gkernel_t, padding=1)
        gxy = F.conv2d(gx, gkernel, padding=1)
        gyy = F.conv2d(gy, gkernel, padding=1)

        hessian = torch.stack([torch.stack([gyy,gxy],dim=-1), torch.stack([gxy,gxx],dim=-1)],dim=-2)
        eig_values, eig_vectors = torch.symeig(hessian, eigenvectors=True)   #Needs PyTorch >=1.6
        max_dir = 1 if max_dir else 0
        hessian_u = eig_vectors[..., :, max_dir] # Min eigen vector
        eig_values = eig_values.abs()
        hessian_vratio = eig_values[..., 1]/eig_values.sum(dim=-1).clamp(10e-7)

        pdir_y, pdir_x = hessian_u[...,0], hessian_u[...,1]
        reverse = 1-2*((pdir_y*gy+pdir_x*gx)<0)
        pdir_y = pdir_y * reverse * hessian_vratio
        pdir_x = pdir_x * reverse * hessian_vratio

        if logs is not None:
            logs['gx'] = gx
            logs['gy'] = gy
            logs['gxx'] = gxx
            logs['gxy'] = gxy
            logs['gyy'] = gyy
            logs['vratio'] = hessian_vratio

        return pdir_y.reshape(b,c,h,w), pdir_x.reshape(b,c,h,w)


def principal_direction(t, smooth_std=5, device=None, hessian_value_threshold=.7):
    """ Compute the principal direction (first eigen vector) of the hessian for each pixels in t.
    t.shape = (b, c, h, w)
    Return vy, vx, vr (namely horizontal and vertical component of the unitary vector of the principal hessian direction, and norm of the vector before normalization).
    """
    with torch.no_grad():
        shape = t.shape
        if len(shape)==2:
            b = 1
            c = 1
            h,w = shape
        elif len(shape)==3:
            c = 1
            b,h,w = shape
        elif len(shape)==4:
            b,c,h,w = shape
        else:
            raise ValueError()

        t = t.reshape(b*c,1,h,w)

        if smooth_std:
            t = smooth(t, smooth_std, device=device)

        gkernel_t = _gkernel_t.to(device) if device else _gkernel_t
        gkernel = _gkernel_t.to(device) if device else _gkernel

        gx = F.conv2d( t, gkernel_t, padding=1)
        gy = F.conv2d( t, gkernel, padding=1)

        if hessian_value_threshold != 1:
            gxx = F.conv2d(gx, gkernel_t, padding=1)
            gxy = F.conv2d(gx, gkernel, padding=1)
            gyy = F.conv2d(gy, gkernel, padding=1)

            hessian = torch.stack([torch.stack([gyy,gxy],dim=-1), torch.stack([gxy,gxx],dim=-1)],dim=-2)
            eig_values, eig_vectors = torch.symeig(hessian, eigenvectors=True)   #Needs PyTorch >=1.6
            hessian_u = eig_vectors[..., :, 1] # Min eigen vector
            eig_values = eig_values.abs()
            hessian_vratio = eig_values[..., 1]/eig_values.sum(dim=-1).clamp(10e-7)

            pdir_y, pdir_x = hessian_u[...,0], hessian_u[...,1]
            reverse = 1-2*((pdir_y*gy+pdir_x*gx)<0)
            pdir_y = pdir_y * reverse * hessian_vratio
            pdir_x = pdir_x * reverse * hessian_vratio

            hessian_mask = hessian_vratio > hessian_value_threshold
            pdir_y = pdir_y*hessian_mask + gy*torch.logical_not(hessian_mask)
            pdir_x = pdir_x*hessian_mask + gx*torch.logical_not(hessian_mask)

        else:
            pdir_y = gy
            pdir_x = gx

        return pdir_y.reshape(*shape), pdir_x.reshape(*shape)