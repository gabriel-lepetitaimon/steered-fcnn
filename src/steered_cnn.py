import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_utils import *
from .rot_utils import radial_steerable_filter


_STEERABLE_FILTERS = tuple(
                        torch.from_numpy([radial_steerable_filter(2*r+1, k, r, .5, sum=1)
                            for k in range(1, (2, 4)[r])])
                        for r in (1, 2))
_RADIAL_FILTERS = tuple(
                    torch.from_numpy([radial_steerable_filter(2*r+1, 0, r, .5, sum=1)])
                    for r in (0, 1, 2))


BaseFilter


class SteeredConv2d(nn.Module):
    def __init__(self, kernel_half_height, n_in, n_out=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 ortho_filters=_STEERABLE_FILTERS, radial_filters=_RADIAL_FILTERS):
        """

        :param kernel_half_height:
        :param n_in:
        :param n_out:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param ortho_filters: A List (of length R) of steerable kernels as pytorch Tensor of shape (K, h, w)
        :param radial_filters: A list (of length R) of radial filters as pytotch Tensor of shape (K, h, w)
        """
        super(SteeredConv2d, self).__init__()
        
        if n_out is None:
            n_out = n_in
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.ortho_filters = ortho_filters
        self.radial_filters = radial_filters

        # Weight
        self.ortho_weights = self.create_weight_from_filters_shape(ortho_filters, ortho=True)
        self.radial_weights = self.create_weight_from_filters_shape(radial_filters, ortho=False)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None

    def create_weight_from_filters_shape(self, filters, ortho=False):
        R = len(filters)
        K = sum([filters[r].shape[0] for r in range(R)])

        if ortho:
            w = nn.Parameter(torch.Tensor(2, self.n_out, self.n_in, K))
            b = np.sqrt(3/(self.n_in*K*2))
        else:
            w = nn.Parameter(torch.Tensor(self.n_out, self.n_in, K))
            b = np.sqrt(3/(self.n_in*K))
        nn.init.uniform_(w, -b, b)
        return w
        
    def forward(self, x, project=None, merge=True):
        # Anti-Symmetric
        asymW = RotConv2d.half2asym(self.asym_half_weight, self.profile)
        u, v = RotConv2d.ortho_conv2d(x, asymW, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, project=project)
        
        # Symmetric
        if self.sym_kernel == 'circ':
            symW  = RotConv2d.half2circ(self.sym_half_weight, self.sym_d_istart)
            padding=self.padding
            if padding=='auto':
                padding = tuple(_//2 for _ in symW.shape[-2:])
            o = F.conv2d(x, symW, stride=self.stride, padding=padding, dilation=self.dilation, groups=self.groups)
        else:
            symW  = RotConv2d.half2sym( self.sym_half_weight, self.profile)
            oy, ox = RotConv2d.ortho_conv2d(x, symW, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            o = torch.cat([oy,ox], 1)
        o = clip_pad_center(o, u.shape)
        # Bias
        if self.u_bias is not None:
            u = u+self.u_bias[None, :, None, None]
            v = v+self.v_bias[None, :, None, None]
            o = o+self.o_bias[None, :, None, None]
        
        if merge:
            return torch.cat([u,v,o], 1)
        else:
            return u, v, o

    def ortho_bases(x, w, stride=1, padding=0, dilation=1, groups=1):
        """
        x: [b,n_in,h,w]
        w: [r][k,H,W] ou [k,H,W]
        return: [b,n_in,r*k,h,w]
        """

        if isinstance(w, (list, tuple)):
            if len(w):
                raise ValueError('The provided steerable kernels list is empty.')
            phi_x, phi_y = SteeredConv2d.ortho_bases(x, w[0],
                                                     stride=stride, padding=padding, dilation=dilation, groups=groups)
            phi
            for W in w[1:]:
                Phi_x, Phi_y = SteeredConv2d.ortho_bases(x, W,
                                                         stride=stride, padding=padding, dilation=dilation, groups=groups)
                phi_x = np.concatenate(clip_tensors(Phi_x, phi_x), axis=0)
                phi_y = np.concatenate(clip_tensors(Phi_y, phi_y), axis=0)
            return phi_x, phi_y

        if padding=='auto':
            hW, wW = w.shape[-2:]
            padding_x = (hW//2, wW//2)
            padding_y = (wW//2, hW//2)
        else:
            padding_x = padding
            padding_y = padding
        phi_x = F.conv2d(x, w[:,None,:,:], stride=stride, padding=padding_x, dilation=dilation, groups=groups)
        w = w.rot90(1, (-2, -1))
        phi_y = F.conv2d(x, w[:,None,:,:], stride=stride, padding=padding_y, dilation=dilation, groups=groups)

        return phi_x, phi_y

    def premultiplied_filters_conv2d(self, x: 'bihw', u: 'UVbjhw'):
        F_x, F_y, F_o = self.composite_filters()
        u_x, u_y = u

        stride = self.stride
        dilation = self.dilation
        groups = self.groups

        padding = SteeredConv2d.get_padding(self.padding, F_x)
        R = u_x * F.conv2d(x, F_x, stride=stride, padding=padding, dilation=dilation, groups=groups)

        padding = SteeredConv2d.get_padding(self.padding, F_y)
        r = u_y * F.conv2d(x, F_y, stride=stride, padding=padding, dilation=dilation, groups=groups)
        R = sum(clip_tensors(R, r))

        padding = SteeredConv2d.get_padding(self.padding, F_o)
        r = F.conv2d(x, F_o, stride=stride, padding=padding, dilation=dilation, groups=groups)
        R = sum(clip_tensors(R, r))

        return R / 3

    def composite_filters(self):
        # --- ORTHOGONAL FILTERS ---
        F_x = None
        F_y = None

        f0 = 0
        W = self.ortho_weights
        for F in self.ortho_filters:
            f1 = f0 + F.shape[0]
            # W: [n_out,n_in,f0:f0+f] x F: [f,h,w] -> [n_out, n_in, h, w]
            # F_x = sum (W_x * f_x - W_y - f_y)
            f =  torch.matmul(W[0,:,:,f0:f1], F)
            f -= torch.matmul(W[1,:,:,f0:f1], F.rot90(1, (1,2)))
            if F_x is None:
                F_x = f
            else:
                F_x = sum(pad_tensors(F_x, f))

            # F_y = -sum (W_y * f_x + W_x - f_y)
            f = -torch.matmul(W[1,:,:,f0:f1], F)
            f -= torch.matmul(W[0,:,:,f0:f1], F.rot90(1, (1,2)))
            if F_y is None:
                F_y = f
            else:
                F_y = sum(pad_tensors(F_y, f))

            f0 = f1

        # --- RADIAL FILTERS ---
        F_o = None
        f0 = 0
        W = self.radial_weights
        for F in self.radial_filters:
            f1 = f0+F.shape[0]
            f = torch.matmul(W[:,:,f0:f1], F)
            F_o = sum(pad_tensors(F_o, f))

            f0 = f1

        return F_x, F_y, F_o

    @staticmethod
    def get_padding(padding, F):
        if padding == 'auto':
            hW, wW = F.shape[-2:]
            return hW//2, wW//2
        return padding

    def _apply(self, fn):
        super(SteeredConv2d, self)._apply(fn)
        self.profile = fn(self.profile)
        if self.sym_d_istart is not None:
            self.sym_d_istart = tuple(fn(_) for _ in self.sym_d_istart)
        return self


class RotConvBN(nn.Module):
    def __init__(self, kernel_half_height, n_in, n_out=None,stride=1, padding=0, dilation=1, groups=1,
                 bn=False, relu=True, squeeze=False, profile='default', sym_kernel='circ'):
        super(RotConvBN, self).__init__()
        
        self._bn = bn
        if n_out is None:
            n_out = n_in
        
        self.conv = RotConv2d(kernel_half_height, n_in, n_out, stride=stride, profile=profile, groups=groups,
                              padding=padding, bias=(not bn and not squeeze), dilation=dilation, sym_kernel=sym_kernel)
        bn_relu = []
        f_out = 3 if sym_kernel=='circ' else 4
        if squeeze:
            bn_relu += [nn.Conv2d(f_out*n_out, n_out, kernel_size=1, bias=not bn)]
            self.n_out = n_out
        else:
            self.n_out = f_out*n_out
        if bn:
            bn_relu += [nn.BatchNorm2d(self.n_out)]
            if relu:
                bn_relu += [nn.ReLU()]
        elif relu:
            bn_relu += [nn.SELU()]

        self.bn_relu = nn.Sequential(*bn_relu)
        
    def forward(self, x, project=None):
        x = self.conv(x,project=project)
        return self.bn_relu(x)
    
    @property
    def bn(self):
        if self._bn:
            return self.model[0]
        return None

    @property
    def relu(self):
        return self.model[1 if self._bn else 0]


_gprof = torch.Tensor([1,2,1])
_gkernel = torch.Tensor([-1,0,1])[:,np.newaxis] * _gprof[np.newaxis,:]
_gkernel_t = _gkernel.transpose(0,1).unsqueeze(0).unsqueeze(0)
_gkernel = _gkernel.unsqueeze(0).unsqueeze(0)


def smooth(t, smooth_std, device=None):
    if isinstance(smooth_std, (int,float)):
        from .gaussian import get_gaussian_kernel2d
        smooth_ksize = int(np.ceil(3*smooth_std))
        smooth_ksize += 1-(smooth_ksize%2)
        smooth_kernel = get_gaussian_kernel2d((smooth_ksize,smooth_ksize),(smooth_std,smooth_std)) \
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