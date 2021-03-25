from torch import nn
import numpy as np
from .torch_utils import *
from .utils import warn
from .reparametrized_cnn import KernelBase


class SteerableOrthoKernelBase(KernelBase):
    def __init__(self, base: 'list(np.array) [n][f,h,w]',
                 base_x: 'list(np.array) [n][2*f,h,w]', base_y: 'list(np.array) [n][2*f,h,w]'):
        super(SteerableOrthoKernelBase, self).__init__(base+base_x+base_y)
        self.R_equi = len(base)
        self.R_ortho = len(base_x)
        assert len(base_x) == len(base_y) and all(bx.shape == by.shape for bx, by in zip(base_x, base_y)), \
            'base_x and base_y should have the same length and shape.'
        self.k_equi = sum(b.shape[0] for b in base)
        self.k_ortho = sum(b.shape[0] for b in base_y)
        self.label = None

    def equi_slice(self):
        return slice(None, self.k_equi)

    def x_slice(self):
        return slice(self.k_equi, self.k_equi+self.k_ortho)

    def y_slice(self):
        return slice(self.k_equi+self.k_ortho, None)

    @property
    def base_equi(self):
        return self.base[self.equi_slice()]

    @property
    def base_x(self):
        return self.base[self.x_slice()]

    @property
    def base_y(self):
        return self.base[self.y_slice()]

    def create_weights(self, n_in, n_out):        
        w = torch.empty((n_out, n_in, self.k_equi))
        b = np.sqrt(3/(n_in*self.k_ortho))
        nn.init.uniform_(w, -b, b)

        w_x = torch.empty((n_out, n_in, self.k_ortho))
        b = np.sqrt(3/(n_in*self.k_ortho*2))
        nn.init.uniform_(w_x, -b, b)

        w_y = torch.empty((n_out, n_in, self.k_ortho))
        b = np.sqrt(3/(n_in*self.k_ortho*2))
        nn.init.uniform_(w_y, -b, b)
        return torch.cat((w, w_x, w_y), dim=2)

    @staticmethod
    def create_from_complex(ortho_base: 'list(np.ndarray) [n][f, h, w]', base: 'list(np.ndarray) [n][f, h, w]' = ()):
        base_x = []
        base_y = []
        for b in ortho_base:
            base_x.append(b.real)
            base_y.append(b.imag)
        return SteerableOrthoKernelBase(base, base_x, base_y)

    @staticmethod
    def create_radial_steerable(rk, std=.5, norm_sum=1):
        """
        rk: - dictionary mapping radius to a list of harmonics count:
                r -> [k0, k1, ...]
            - int: r_max
        return: A list of steerable filters with shape [r][k,h,w].
        """
        if isinstance(rk, int):
            R = rk
            rk = {}

            def circle_area(radius):
                return np.pi * radius ** 2

            def max_k(radius):
                inter_area = circle_area(radius + .5) - circle_area(radius - .5)
                return int(inter_area//2)

            for r in range(R):
                rk[r] = list(range(max_k(r)))

        from .rot_utils import polar_space
        K_ortho = []
        K_equi = []
        label_ortho = []
        label_equi = []
        for r, Ks in rk.items():
            if r == 0:
                K_equi.append(np.ones((1, 1, 1)))
                label_equi.append('r0')
            else:
                size = r*2+1
                rho, phi = polar_space(size)
                G = np.exp(-(rho-r)**2/(2 * std**2))

                kernels = []
                for k in Ks:
                    g = G[:]
                    if k == 0:
                        kernel = g
                        if norm_sum:
                            kernel = kernel / kernel.sum() * norm_sum
                        K_equi.append(kernel[None, :, :])
                        label_equi.append(f'r{r}k{k}')
                    else:
                        if k == 0:
                            g[rho == 0] *= 0
                        PHI = np.exp(1j*k*phi)
                        kernel = g*PHI
                        if norm_sum:
                            kernel = kernel / kernel.sum() * norm_sum
                        kernels.append(kernel)
                        label_ortho.append(f'r{r}k{k}')
                if kernels:
                    K_ortho.append(np.stack(kernels))
        B = SteerableOrthoKernelBase.create_from_complex(K_ortho, K_equi)
        B.label = label_equi + [_+'x' for _ in label_ortho] + [_+'y' for _ in label_ortho]
        return B

    def conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [2, n_out, n_in, h, w]',
               project: 'torch.Tensor [UV,b,j,h,w]' = None, precompute_kernel=None,
               stride=1, padding='auto', dilation=1):
        if self.device != input.device:
            self.to(input.device)

        if precompute_kernel is None:
            precompute_kernel = True

        if precompute_kernel:
            f, fx, fy = self.composite_kernels_conv2d(input=input, weight=weight, stride=stride,
                                                      padding=padding, dilation=dilation)
        else:
            f, fx, fy = self.preconvolved_base_conv2d(input=input, weight=weight, stride=stride,
                                                      padding=padding, dilation=dilation)

        if project is None:
            return f, fx, fy
        else:
            u, v = project
            if u.shape[-1] / fx.shape[-1] > 2 and u.shape[-1] / fx.shape[-1] > 2:
                warn(f'WARNING: During the computation of a steered neuron, the dimension of the provided project '
                     f'field is extensively larger than the dimension of the features. '
                     f'It might be a down-sampling implementation error.\n'
                     f'project.shape: {u.shape}, f.shape: {fx.shape}')
            f += clip_pad_center(u, fx.shape) * fx
            f += clip_pad_center(v, fy.shape) * fy
            return f

    # --- Composite Kernels ---
    def composite_ortho_kernels(self, weight: 'np.array [n_out, n_in, K+Kx+Ky]'):
        k0 = 0
        W = weight[..., self.equi_slice()]
        Wx, Wy = weight[..., self.x_slice()], weight[..., self.y_slice()]
        n_out, n_in, n_k = Wx.shape
        k, h_k, w_k = self.base[0].shape

        K_x = torch.zeros((n_out, n_in, h_k, w_k), device=self.base[0].device)
        K_y = torch.zeros((n_out, n_in, h_k, w_k), device=self.base[0].device)

        for base_x, base_y in zip(self.base_x, self.base_y):
            k, h_k, w_k = base_x.shape
            bx = base_x.reshape(k, h_k*w_k)
            by = base_y.reshape(k, h_k*w_k)
            k1 = k0 + k

            # W: [n_out,n_in,k0:k0+k] x K: [k,h,w] -> [n_out, n_in, h, w]
            # F_x = sum (Wx * Kx - Wy - Ky)
            print(Wx[:, :, k0:k1].shape, bx.shape)
            kernel = torch.matmul(Wx[:, :, k0:k1], bx).reshape(n_out, n_in, h_k, w_k)
            kernel -= torch.matmul(Wy[:, :, k0:k1], by).reshape(n_out, n_in, h_k, w_k)
            K_x = sum(pad_tensors(K_x, kernel))

            # F_y = -sum (W_y * f_x + W_x - f_y)
            kernel = torch.matmul(Wy[:, :, k0:k1], bx).reshape(n_out, n_in, h_k, w_k)
            kernel += torch.matmul(Wx[:, :, k0:k1], by).reshape(n_out, n_in, h_k, w_k)
            K_y = sum(pad_tensors(K_y, kernel))

            k0 = k1
        return KernelBase.composite_kernels(self.base_equi, W), K_x, K_y

    def composite_kernels_conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [2, n_out, n_in, h, w]',
                                 stride=1, padding='auto', dilation=1, groups=1):
        W, Wx, Wy = self.composite_ortho_kernels(weight)

        paddingX = get_padding(padding, W.shape)
        f = F.conv2d(input, W, stride=stride, padding=paddingX, dilation=dilation, groups=groups)

        paddingX = get_padding(padding, Wx.shape)
        fx = F.conv2d(input, Wx, stride=stride, padding=paddingX, dilation=dilation, groups=groups)

        paddingY = get_padding(padding, Wy.shape)
        fy = F.conv2d(input, Wy, stride=stride, padding=paddingY, dilation=dilation, groups=groups)

        return f, fx, fy

    # --- Preconvolve Bases ---
    def preconvolve_ortho_bases(self, input: 'torch.Tensor [b,i,w,h]', stride=1, padding='auto', dilation=1):
        base  = KernelBase.preconvolve_base(input, self.base_equi, stride=stride, padding=padding, dilation=dilation)
        baseX = KernelBase.preconvolve_base(input, self.base_x, stride=stride, padding=padding, dilation=dilation)
        baseY = KernelBase.preconvolve_base(input, self.base_y, stride=stride, padding=padding, dilation=dilation)
        return base, baseX, baseY

    def preconvolved_base_conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [2, n_out, n_in, k]',
                                 stride=1, padding='auto', dilation=1):
        base, baseX, baseY = self.preconvolve_ortho_bases(input,
                                                          stride=stride, padding=padding, dilation=dilation)
        b, n_in, k, h, w = base.shape
        n_out, n_in_w, k_w = weight.shape

        assert n_in == n_in_w, f"The provided inputs and weights have different number of input neuron:\n " + \
                               f"x.shape[1]=={n_in}, weigth.shape[1]=={n_in_w}."
        assert 2*self.k_ortho+self.k_equi == k_w, f"The provided weights have an incorrect number of kernels:\n " + \
                                                  f"weight.shape[2]=={k_w}, but should be {2*self.k_ortho+self.k_equi}."

        K = base.permute(0, 3, 4, 1, 2).reshape(b, h, w, n_in*self.k_equi)
        W = weight[..., self.equi_slice()].reshape(n_out, n_in*self.k_equi).transpose(0, 1)
        f = torch.matmul(K, W).permute(0, 3, 1, 2)
        K, W = None, None

        Kx = baseX.permute(0, 3, 4, 1, 2).reshape(b, h, w, n_in*self.k_ortho)
        Ky = baseY.permute(0, 3, 4, 1, 2).reshape(b, h, w, n_in*self.k_ortho)
        Wx = weight[..., self.x_slice()].reshape(n_out, n_in*self.k_ortho).transpose(0, 1)
        Wy = weight[..., self.y_slice()].reshape(n_out, n_in**self.k_ortho).transpose(0, 1)

        # K:[b,h,w,n_in*k] x W:[n_in*k, n_out] -> [b,h,w,n_out]
        fx = (torch.matmul(Kx, Wx) - torch.matmul(Ky, Wy)).permute(0, 3, 1, 2)  # [b,n_out,h,w]
        fy = (torch.matmul(Kx, Wy) + torch.matmul(Ky, Wx)).permute(0, 3, 1, 2)

        return f, fx, fy


_DEFAULT_STEERABLE_BASE = SteerableOrthoKernelBase.create_radial_steerable(4)


class SteeredConv2d(nn.Module):
    def __init__(self, n_in, n_out=None, steerable_base: SteerableOrthoKernelBase = None,
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
            steerable_base = SteerableOrthoKernelBase.create_radial_steerable(steerable_base)
        self.base = steerable_base

        # Weight
        self.weights = nn.Parameter(self.base.create_weights(n_in, n_out), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_out), requires_grad=True) if bias else None

    def forward(self, x, project):
        out = self.base.conv2d(x, self.weights, project=project,
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
        
    def forward(self, x, project=None):
        x = self.conv(x, project=project)
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