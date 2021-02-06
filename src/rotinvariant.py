import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchgeometry.image as TGI


def clip_pad_center(tensor, shape, pad_mode='constant', pad_value=0):
    s = tensor.shape[-2:] 
    
    y0 = (s[0]-shape[-2])//2
    y1 = 0
    if y0 < 0:
        y1 = -y0
        y0 = 0
        
    x0 = (s[1]-shape[-1])//2
    x1 = 0
    if x0 < 0:
        x1 = -x0
        x0 = 0
    tensor = tensor[..., y0:y0+shape[-2], x0:x0+shape[-1]]
    if x1 or y1:
        tensor = F.pad(tensor, (y1,y1,x1,x1), mode=pad_mode, value=pad_value)
    return tensor


def clip_tensors(t1, t2):
    if t1.shape[-2:] == t2.shape[-2:]:
        return t1, t2
    h1, w1 = t1.shape[-2:]
    h2, w2 = t2.shape[-2:]
    dh = h1-h2
    dw = w1-w2
    i1 = max(dh,0)
    j1 = max(dw,0)
    h = h1 - i1
    w = w1 - j1
    i1 = i1 // 2
    j1 = j1 // 2
    i2 = i1 - dh//2
    j2 = j1 - dw//2
    
    t1 = t1[...,i1:i1+h, j1:j1+w]
    t2 = t2[...,i2:i2+h, j2:j2+w]
    return t1, t2

_gprof = torch.Tensor([1,2,1])
_gkernel = torch.Tensor([-1,0,1])[:,np.newaxis] * _gprof[np.newaxis,:]
_gkernel_t = _gkernel.transpose(0,1).unsqueeze(0).unsqueeze(0)
_gkernel = _gkernel.unsqueeze(0).unsqueeze(0)


def smooth(t, smooth_std, device=None):
    if isinstance(smooth_std, (int,float)):
        smooth_ksize = int(np.ceil(3*smooth_std))
        smooth_ksize += 1-(smooth_ksize%2)
        smooth_kernel = TGI.get_gaussian_kernel2d((smooth_ksize,smooth_ksize),(smooth_std,smooth_std))\
                           .unsqueeze(0).unsqueeze(0)
        if device is not None:
            smooth_kernel = smooth_kernel.to(device)
            
    elif isinstance(smooth_std, torch.Tensor):
        smooth_kernel = smooth_std.device(device) if device is not None else smooth_std
        smooth_smooth_kernel.unsqueeze(0).unsqueeze(0)
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


class RotConv2d(nn.Module):
    def __init__(self, kernel_half_height, n_in, n_out=None,
                 stride=1, padding=0, dilation=1, groups=1,  bias=True, 
                 sym_kernel='circ', profile='default'):
        super(RotConv2d, self).__init__()
        
        if n_out is None:
            n_out = n_in
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.sym_kernel = sym_kernel
        
        # Weight
        self.asym_half_weight = nn.Parameter(torch.Tensor(n_out, n_in, kernel_half_height))
        self.sym_half_weight = nn.Parameter(torch.Tensor(n_out, n_in, kernel_half_height))
        b = np.sqrt(3/(n_in*(2*kernel_half_height)))   # No ReLu
        nn.init.uniform_(self.asym_half_weight, -b, b)
        nn.init.uniform_(self.sym_half_weight, -b, b)
        
        #Bias
        self.u_bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.v_bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.o_bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        
        # Profile
        if profile == 'default':
            profile = kernel_half_height
        if not profile:
            self.profile = None
        elif isinstance(profile, int):
            self.profile = RotConv2d.get_profile(profile)
        elif isinstance(profile, (tuple,list)):
            self.profile = torch.tensor(profile, req)
        else:
            self.profile = profile
        if sym_kernel=='circ':
            self.sym_d_istart = RotConv2d.d_istart(kernel_half_height, odd=False)
        else:
            self.sym_d_istart = None
        
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
            o = torch.stack([oy,ox], 0).norm(dim=0)
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
    
    @staticmethod
    def half2sym(half_weight, profile=None):
        """
        half_weight.shape = (n_out, n_in, half_height)
        return shape: (n_out, n_in, half_height*2, len(profile)) (or 1 if profile is None)
        """
        l = torch.cat((half_weight.flip(2), half_weight), 2)
        l = l.unsqueeze(3)
        if profile is not None:
            return l*profile[np.newaxis,np.newaxis,np.newaxis,:]
        return l
    
    @staticmethod
    def d_istart(shape, odd=True, device=None):
        s = shape
        if odd:
            xy = torch.linspace(-s, s, s*2+1, device=device)
        else:
            xy = torch.linspace(-s+0.5, s-0.5, s*2, device=device)
        xx, yy = torch.meshgrid(xy, xy)
        d = torch.stack((xx,yy), 0).norm(dim=0)
        istart = torch.floor(d)
        d = d - istart
        return d, istart.long()
    
    @staticmethod
    def half2circ(half_weight, d_istart=None):
        """
        half_weight.shape = (n_out, n_in, half_height)
        return shape: (n_out, n_in, half_height*2, half_height*2)
        """            
        w = F.pad(half_weight, (0,1))
            
        s = half_weight.shape[-1]
        if d_istart is None:
            d, istart = d_istart(s, device=half_weight.device)
        else:
            d, istart = d_istart
        kernel = (1-d)*w[..., torch.clamp(istart,max=s)] + d*w[..., torch.clamp(istart,max=s)]
        return kernel
        
    
    @staticmethod
    def half2asym(half_weight, profile=None):
        """
        half_weight.shape = (n_out, n_in, half_height)
        return shape: (n_out, n_in, half_height*2, len(profile)) (or 1 if profile is None)
        """
        l = torch.cat((-half_weight.flip(2), half_weight), 2)
        l = l.unsqueeze(3)
        if profile is not None:
            return l*profile[np.newaxis,np.newaxis,np.newaxis,:]
        return l
    
    @staticmethod
    def get_profile(l,device=None):
        p = torch.cumsum(torch.ones((l,),requires_grad=False,device=None),0)
        return torch.cat((p,p.flip(0))) / (p.sum()*2)
    
    @staticmethod
    def ortho_conv2d(x, w, stride=1, padding=0, dilation=1, groups=1, project=None):
        """
        x[b,f,y,x]
        w[f_out,f_in,u,v]
        project.shape = (b, 2, h, w)
        """
        if padding=='auto':
            hW, wW = w.shape[-2:]
            padding_x = (hW//2, wW//2)
            padding_y = (wW//2, hW//2)
        else:
            padding_x = padding
            padding_y = padding
        r_x = F.conv2d(x, w, stride=stride, padding=padding_x, dilation=dilation, groups=groups)
        w = w.rot90(1, (2,3))
        r_y = F.conv2d(x, w, stride=stride, padding=padding_y, dilation=dilation, groups=groups)
        
        if project is not None:
            p_y, p_x = project
            if p_y.ndim==3:
                p_y = p_y.unsqueeze(1)
                p_x = p_x.unsqueeze(1)
            
            p_y, r_y = clip_tensors(p_y, r_y)
            p_x, r_x = clip_tensors(p_x, r_x)
            r_u =  sum(clip_tensors(p_y*r_y, p_x*r_x))
            
            p_y, r_x = clip_tensors(p_y, r_x)
            p_x, r_y = clip_tensors(p_x, r_y)
            r_v = sum(clip_tensors(-p_x*r_y, p_y*r_x))
            
            r_u, r_v = clip_tensors(r_u, r_v)
            return r_u, r_v
        else:
            return r_x, r_y
        
    def _apply(self, fn):
        super(RotConv2d, self)._apply(fn)
        self.profile = fn(self.profile)
        if self.sym_d_istart is not None:
            self.sym_d_istart = tuple(fn(_) for _ in self.sym_d_istart)
        return self


class RotConvBN(nn.Module):
    def __init__(self, kernel_half_height, n_in, n_out=None,stride=1, padding=0, dilation=1, groups=1, 
                 bn=False, relu=True, squeeze=False, profile='default'):
        super(RotConvBN, self).__init__()
        
        self._bn = bn
        if n_out is None:
            n_out = n_in
        
        self.conv = RotConv2d(kernel_half_height, n_in, n_out, stride=stride, 
                              padding=padding, bias=(not bn and not squeeze), dilation=dilation)
        bn_relu = []
        if squeeze:
            bn_relu += [nn.Conv2d(3*n_out,n_out,kernel_size=1, bias=not bn)]
        if bn:
            bn_relu += [nn.BatchNorm2d(n_out*(1 if squeeze else 3))]
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
