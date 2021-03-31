import torch
from torch import nn
import torch.nn.functional as F

from .torch_utils import *
from .steered_cnn import SteeredConvBN, SteerableKernelBase
from .steered_cnn import principal_direction as compute_pdir


class HemelingNet(nn.Module):
    def __init__(self, n_in, n_out=1, p_dropout=0, nfeatures_base=16, half_kernel_height=3, padding=0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16
        
        kernel_height = half_kernel_height*2-1
        
        # Down
        self.conv1 = ConvBN(kernel_height, n_in, n1, relu=True, padding=padding)
        self.conv2 = ConvBN(kernel_height, n1, n1, relu=True, padding=padding)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = ConvBN(kernel_height, n1, n2, relu=True, padding=padding)
        self.conv4 = ConvBN(kernel_height, n2, n2, relu=True, padding=padding)        
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = ConvBN(kernel_height, n2, n3, relu=True, padding=padding)
        self.conv6 = ConvBN(kernel_height, n3, n3, relu=True, padding=padding)        
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv7 = ConvBN(kernel_height, n3, n4, relu=True, padding=padding)
        self.conv8 = ConvBN(kernel_height, n4, n4, relu=True, padding=padding)        
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv9 = ConvBN(kernel_height, n4, n5, relu=True, padding=padding)
        self.conv10 = ConvBN(kernel_height, n5, n5, relu=True, padding=padding)        
        
        # Up
        self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=2, stride=2)
        self.conv11 = ConvBN(kernel_height, n5, n4, relu=True, padding=padding)
        self.conv12 = ConvBN(kernel_height, n4, n4, relu=True, padding=padding)        
        
        self.upsample2 = nn.ConvTranspose2d(n4,n3, kernel_size=2, stride=2)
        self.conv13 = ConvBN(kernel_height, n4, n3, relu=True, padding=padding)
        self.conv14 = ConvBN(kernel_height, n3, n3, relu=True, padding=padding)        
        
        self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=2, stride=2)
        self.conv15 = ConvBN(kernel_height, n3, n2, relu=True, padding=padding)
        self.conv16 = ConvBN(kernel_height, n2, n2, relu=True, padding=padding)        
        
        self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=2, stride=2)
        self.conv17 = ConvBN(kernel_height, n2, n1, relu=True, padding=padding)
        self.conv18 = ConvBN(kernel_height, n1, n1, relu=True, padding=padding)        

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=1)
        
        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else lambda x: x

    def forward(self, x, **kwargs):
                
        # Down
        x1 = self.conv2(self.conv1(x))
        
        x2 = self.pool1(x1)
        x2 = self.conv4(self.conv3(x2))
        
        x3 = self.pool2(x2)
        x3 = self.conv6(self.conv5(x3))
        
        x4 = self.pool3(x3)
        x4 = self.conv8(self.conv7(x4))
        
        x = self.pool4(x4)
        x = self.dropout(self.conv9(x))
        x = self.dropout(self.conv10(x))
        
        # Up
        x4 = cat_crop(x4, self.upsample1(x))
        x4 = self.conv12(self.conv11(x4))
        
        x3 = cat_crop(x3, self.upsample2(x4))
        x3 = self.conv14(self.conv13(x3))
        
        x2 = cat_crop(x2, self.upsample3(x3))
        x2 = self.conv16(self.conv15(x2))
        
        x1 = cat_crop(x1, self.upsample4(x2))
        x1 = self.conv18(self.conv17(x1))
        
        # End
        return self.final_conv(x1)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p


class HemelingRotNet(nn.Module):

    def __init__(self, n_in, n_out=1, nfeatures_base=6, depth=2, base=None,
                 p_dropout=0, padding=0,
                 static_principal_direction=False,
                 principal_direction='all', principal_direction_smooth=3, principal_direction_hessian_threshold=1):
        super(HemelingRotNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.static_principal_direction = static_principal_direction
        self.principal_direction = principal_direction
        self.principal_direction_smooth = principal_direction_smooth
        self.principal_direction_hessian_threshold = principal_direction_hessian_threshold
        
        if base is None:
            base = SteerableKernelBase.create_from_rk(4, max_k=5)
        elif isinstance(steerable_base, (int, dict)):
            base = SteerableKernelBase.create_from_rk(base)
        self.base = base
        
        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16
        
        # Down
        self.conv1 = nn.ModuleList(
                     [SteeredConvBN(n_in, n1, relu=True, bn=True, padding=padding, steerable_base=base)]
                   + [SteeredConvBN(n1, n1, relu=True, bn=True, padding=padding, steerable_base=base)
                      for i in range(depth-1)])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.ModuleList(
                     [SteeredConvBN(n1, n2, relu=True, bn=True, padding=padding, steerable_base=base)]
                   + [SteeredConvBN(n2, n2, relu=True, bn=True, padding=padding, steerable_base=base)
                      for i in range(depth-1)])
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.ModuleList(
                     [SteeredConvBN(n2, n3, relu=True, bn=True,   padding=padding, steerable_base=base)]
                   + [SteeredConvBN(n3, n3, relu=True, bn=True,   padding=padding, steerable_base=base)
                      for i in range(depth-1)])
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.ModuleList(
                     [SteeredConvBN(n3, n4, relu=True, bn=True,   padding=padding, steerable_base=base)]
                   + [SteeredConvBN(n4, n4, relu=True, bn=True,   padding=padding, steerable_base=base)
                      for i in range(depth-1)])
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.ModuleList(
                     [SteeredConvBN(n4, n5, relu=True, bn=True, padding=padding, steerable_base=base)]
                   + [SteeredConvBN(n5, n5, relu=True, bn=True, padding=padding, steerable_base=base)
                      for i in range(depth-1)])
        
        # Up
        self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=2, stride=2)
        self.conv6 = nn.ModuleList(
                     [SteeredConvBN(2*n4, n4, relu=True, bn=True, padding=padding, steerable_base=base)]
                   + [SteeredConvBN(  n4, n4, relu=True, bn=True, padding=padding, steerable_base=base)
                      for i in range(depth-1)])
        
        self.upsample2 = nn.ConvTranspose2d(n4, n3, kernel_size=2, stride=2)
        self.conv7 = nn.ModuleList(
                     [SteeredConvBN(2*n3, n3, relu=True, bn=True, padding=padding, steerable_base=base)]
                   + [SteeredConvBN(  n3, n3, relu=True, bn=True, padding=padding, steerable_base=base)
                      for i in range(depth-1)])
        
        self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=2, stride=2)
        self.conv8 = nn.ModuleList(
                     [SteeredConvBN(2*n2, n2, relu=True, bn=True, padding=padding, steerable_base=base)]
                   + [SteeredConvBN(  n2, n2, relu=True, bn=True, padding=padding, steerable_base=base)
                      for i in range(depth-1)])

        self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=2, stride=2)
        self.conv9 = nn.ModuleList(
                     [SteeredConvBN(2*n1, n1, relu=True, bn=True, padding=padding, steerable_base=base)]
                   + [SteeredConvBN(  n1, n1, relu=True, bn=True, padding=padding, steerable_base=base)
                      for i in range(depth-1)])

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=1)
        
        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else lambda x: x

    def forward(self, x, alpha=None, **kwargs):
        from functools import reduce
        if alpha is None:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                if alpha.dim == 4 and alpha.shape[1] == 2:
                    alpha = torch.atan2(alpha[:, 1], alpha[:, 0])
                
                max_k = self.base.max_k
                b, h, w = alpha.shape
                
                k_alpha = torch.stack([k*alpha for k in range(1, self.base.max_k+1)])
                cos_sin_kalpha = torch.stack((torch.cos(k_alpha), torch.sin(k_alpha)))
                
                alpha1 = cos_sin_kalpha.reshape((2*max_k,b,h,w))
                alpha2 = F.avg_pool2d(alpha1, 2)
                alpha3 = F.avg_pool2d(alpha2, 2)
                alpha4 = F.avg_pool2d(alpha3, 2)
                alpha5 = F.avg_pool2d(alpha4, 2)

                alpha1 = tuple(alpha1.reshape(2,max_k,b,1, *alpha1.shape[-2:]))
                alpha2 = tuple(alpha2.reshape(2,max_k,b,1, *alpha2.shape[-2:]))
                alpha3 = tuple(alpha3.reshape(2,max_k,b,1, *alpha3.shape[-2:]))
                alpha4 = tuple(alpha4.reshape(2,max_k,b,1, *alpha4.shape[-2:]))
                alpha5 = tuple(alpha5.reshape(2,max_k,b,1, *alpha5.shape[-2:]))

        # Down
        x1 = reduce(lambda x, conv: conv(x, alpha=alpha1), self.conv1, x)

        x2 = self.pool1(x1)
        x2 = reduce(lambda x, conv: conv(x, alpha=alpha2), self.conv2, x2)

        x3 = self.pool2(x2)
        x3 = reduce(lambda x, conv: conv(x, alpha=alpha3), self.conv3, x3)
        
        x4 = self.pool3(x3)
        x4 = reduce(lambda x, conv: conv(x, alpha=alpha4), self.conv4, x4)
        
        x5 = self.pool4(x4)
        x5 = reduce(lambda x, conv: conv(x, alpha=alpha5), self.conv5, x5)
        x5 = self.dropout(x5)
        
        # Up
        x4 = cat_crop(x4, self.upsample1(x5))
        x5 = None
        x4 = reduce(lambda x, conv: conv(x, alpha=alpha4), self.conv6, x4)
        
        x3 = cat_crop(x3, self.upsample2(x4))
        x4 = None
        x3 = reduce(lambda x, conv: conv(x, alpha=alpha3), self.conv7, x3)
        
        x2 = cat_crop(x2, self.upsample3(x3))
        x3 = None
        x2 = reduce(lambda x, conv: conv(x, alpha=alpha2), self.conv8, x2)
        
        x1 = cat_crop(x1, self.upsample4(x2))
        x2 = None
        x1 = reduce(lambda x, conv: conv(x, alpha=alpha1), self.conv9, x1)
        
        # End
        return self.final_conv(x1)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p