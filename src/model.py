import torch
from torch import nn
import torch.nn.functional as F

from .torch_utils import *
from .steered_cnn import SteeredConvBN
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

    def __init__(self, n_in, n_out=1, nfeatures_base=6, half_kernel_height=3, depth=2,
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
        
        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16
        
        # Down
        self.conv1 = nn.ModuleList(
                     [SteeredConvBN(n_in, n1, relu=True, bn=True, padding=padding)]
                   + [SteeredConvBN(n1, n1, relu=True, bn=True, padding=padding)
                      for i in range(depth-1)])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.ModuleList(
                     [SteeredConvBN(n1, n2, relu=True, bn=True, padding=padding)]
                   + [SteeredConvBN(n2, n2, relu=True, bn=True, padding=padding)
                      for i in range(depth-1)])
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.ModuleList(
                     [SteeredConvBN(n2, n3, relu=True, bn=True,   padding=padding)]
                   + [SteeredConvBN(n3, n3, relu=True, bn=True,   padding=padding)
                      for i in range(depth-1)])
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.ModuleList(
                     [SteeredConvBN(n3, n4, relu=True, bn=True,   padding=padding)]
                   + [SteeredConvBN(n4, n4, relu=True, bn=True,   padding=padding)
                      for i in range(depth-1)])
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.ModuleList(
                     [SteeredConvBN(n4, n5, relu=True, bn=True, padding=padding)]
                   + [SteeredConvBN(n5, n5, relu=True, bn=True, padding=padding)
                      for i in range(depth-1)])
        
        # Up
        self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=2, stride=2)
        self.conv6 = nn.ModuleList(
                     [SteeredConvBN(2*n4, n4, relu=True, bn=True, padding=padding)]
                   + [SteeredConvBN(  n4, n4, relu=True, bn=True, padding=padding)
                      for i in range(depth-1)])
        
        self.upsample2 = nn.ConvTranspose2d(n4, n3, kernel_size=2, stride=2)
        self.conv7 = nn.ModuleList(
                     [SteeredConvBN(2*n3, n3, relu=True, bn=True, padding=padding)]
                   + [SteeredConvBN(  n3, n3, relu=True, bn=True, padding=padding)
                      for i in range(depth-1)])
        
        self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=2, stride=2)
        self.conv8 = nn.ModuleList(
                     [SteeredConvBN(2*n2, n2, relu=True, bn=True, padding=padding)]
                   + [SteeredConvBN(  n2, n2, relu=True, bn=True, padding=padding)
                      for i in range(depth-1)])

        self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=2, stride=2)
        self.conv9 = nn.ModuleList(
                     [SteeredConvBN(2*n1, n1, relu=True, bn=True, padding=padding)]
                   + [SteeredConvBN(  n1, n1, relu=True, bn=True, padding=padding)
                      for i in range(depth-1)])

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=1)
        
        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else lambda x: x

    def forward(self, x, principal_direction=None, **kwargs):
        from functools import reduce
        if not self.static_principal_direction or principal_direction is None:
            if self.principal_direction == 'all':
                downSampledX = x.mean(axis=1)
            else:
                downSampledX = x[:, self.principal_direction]
            device = self.final_conv.weight.device
            pDir1 = compute_pdir(downSampledX, device=device,
                                 hessian_value_threshold=self.principal_direction_hessian_threshold,
                                 smooth_std=self.principal_direction_smooth)[:, None, :, :]
            downSampledX = F.avg_pool2d(downSampledX, 2)
            pDir2 = compute_pdir(downSampledX, device=device,
                                 hessian_value_threshold=self.principal_direction_hessian_threshold,
                                 smooth_std=self.principal_direction_smooth)[:, None, :, :]
            downSampledX = F.avg_pool2d(downSampledX, 2)
            pDir3 = compute_pdir(downSampledX, device=device,
                                 hessian_value_threshold=self.principal_direction_hessian_threshold,
                                 smooth_std=self.principal_direction_smooth)[:, None, :, :]
            downSampledX = F.avg_pool2d(downSampledX, 2)
            pDir4 = compute_pdir(downSampledX, device=device,
                                 hessian_value_threshold=self.principal_direction_hessian_threshold,
                                 smooth_std=self.principal_direction_smooth)[:, None, :, :]
            downSampledX = F.avg_pool2d(downSampledX, 2)
            pDir5 = compute_pdir(downSampledX, device=device,
                                 hessian_value_threshold=self.principal_direction_hessian_threshold,
                                 smooth_std=self.principal_direction_smooth)[:, None, :, :]
        else:
            pDir1 = principal_direction.transpose(1, 0)
            pDir2 = F.avg_pool2d(pDir1, 2)
            pDir3 = F.avg_pool2d(pDir2, 2)
            pDir4 = F.avg_pool2d(pDir3, 2)
            pDir5 = F.avg_pool2d(pDir4, 2)

            pDir1 = pDir1[:, :, None, :, :]
            pDir2 = pDir2[:, :, None, :, :]
            pDir3 = pDir3[:, :, None, :, :]
            pDir4 = pDir4[:, :, None, :, :]
            pDir5 = pDir5[:, :, None, :, :]

        # Down
        x1 = reduce(lambda x, conv: conv(x, project=pDir1), self.conv1, x)

        x2 = self.pool1(x1)
        x2 = reduce(lambda x, conv: conv(x, project=pDir2), self.conv2, x2)

        x3 = self.pool2(x2)
        x3 = reduce(lambda x, conv: conv(x, project=pDir3), self.conv3, x3)
        
        x4 = self.pool3(x3)
        x4 = reduce(lambda x, conv: conv(x, project=pDir4), self.conv4, x4)
        
        x5 = self.pool4(x4)
        x5 = reduce(lambda x, conv: conv(x, project=pDir5), self.conv5, x5)
        x5 = self.dropout(x5)
        
        # Up
        x4 = cat_crop(x4, self.upsample1(x5))
        x5 = None
        x4 = reduce(lambda x, conv: conv(x, project=pDir4), self.conv6, x4)
        
        x3 = cat_crop(x3, self.upsample2(x4))
        x4 = None
        x3 = reduce(lambda x, conv: conv(x, project=pDir3), self.conv7, x3)
        
        x2 = cat_crop(x2, self.upsample3(x3))
        x3 = None
        x2 = reduce(lambda x, conv: conv(x, project=pDir2), self.conv8, x2)
        
        x1 = cat_crop(x1, self.upsample4(x2))
        x2 = None
        x1 = reduce(lambda x, conv: conv(x, project=pDir1), self.conv9, x1)
        
        # End
        return self.final_conv(x1)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p