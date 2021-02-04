import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .rotinvariant import RotConvBN, principal_direction


class HemelingNet(nn.Module):
    def __init__(self, n_in, n_out=1, p_dropout=0, nfeatures_base=16, half_kernel_height=3):
        super(HemelingNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        
        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16
        
        kernel_height = half_kernel_height*2-1
        
        # Down
        self.conv1 = ConvBN(kernel_height, n_in, n1, relu=True)
        self.conv2 = ConvBN(kernel_height, n1, n1, relu=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = ConvBN(kernel_height, n1, n2, relu=True)
        self.conv4 = ConvBN(kernel_height, n2, n2, relu=True)        
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = ConvBN(kernel_height, n2, n3, relu=True)
        self.conv6 = ConvBN(kernel_height, n3, n3, relu=True)        
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv7 = ConvBN(kernel_height, n3, n4, relu=True)
        self.conv8 = ConvBN(kernel_height, n4, n4, relu=True)        
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv9 = ConvBN(kernel_height, n4, n5, relu=True)
        self.conv10 = ConvBN(kernel_height, n5, n5, relu=True)        
        
        # Up
        self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=2, stride=2)
        self.conv11 = ConvBN(kernel_height, n5, n4, relu=True)
        self.conv12 = ConvBN(kernel_height, n4, n4, relu=True)        
        
        self.upsample2 = nn.ConvTranspose2d(n4,n3, kernel_size=2, stride=2)
        self.conv13 = ConvBN(kernel_height, n4, n3, relu=True)
        self.conv14 = ConvBN(kernel_height, n3, n3, relu=True)        
        
        self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=2, stride=2)
        self.conv15 = ConvBN(kernel_height, n3, n2, relu=True)
        self.conv16 = ConvBN(kernel_height, n2, n2, relu=True)        
        
        self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=2, stride=2)
        self.conv17 = ConvBN(kernel_height, n2, n1, relu=True)
        self.conv18 = ConvBN(kernel_height, n1, n1, relu=True)        

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=1)
        
        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else lambda x: x

    def forward(self, x):
                
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


class HemelingRotNet(nn.Module):
    def __init__(self, n_in, n_out=1, nfeatures_base=6, half_kernel_height=3,
                 p_dropout=0, rotconv_squeeze=False,
                 principal_direction='all', principal_direction_smooth=3, principal_direction_hessian_threshold=1):
        super(HemelingRotNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.principal_direction = principal_direction
        self.principal_direction_smooth = principal_direction_smooth
        self.principal_direction_hessian_threshold = principal_direction_hessian_threshold
        
        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16
        
        o = 1 if rotconv_squeeze else 3
        
        # Down
        self.conv1 = RotConvBN(half_kernel_height, n_in, n1, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv2 = RotConvBN(half_kernel_height, o*n1, n1, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = RotConvBN(half_kernel_height, o*n1, n2, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv4 = RotConvBN(half_kernel_height, o*n2, n2, relu=True, bn=True, squeeze=rotconv_squeeze)        
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = RotConvBN(half_kernel_height, o*n2, n3, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv6 = RotConvBN(half_kernel_height, o*n3, n3, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv7 = RotConvBN(half_kernel_height, o*n3, n4, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv8 = RotConvBN(half_kernel_height, o*n4, n4, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv9 = RotConvBN(half_kernel_height, o*n4, n5, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv10 = RotConvBN(half_kernel_height,o*n5, n5, relu=True, bn=True, squeeze=rotconv_squeeze)
        
        # Up
        self.upsample1 = nn.ConvTranspose2d(o*n5,o*n4, kernel_size=2, stride=2)
        self.conv11 = RotConvBN(half_kernel_height, 2*o*n4, n4, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv12 = RotConvBN(half_kernel_height,   o*n4, n4, relu=True, bn=True, squeeze=rotconv_squeeze)
        
        self.upsample2 = nn.ConvTranspose2d(o*n4,o*n3, kernel_size=2, stride=2)
        self.conv13 = RotConvBN(half_kernel_height, 2*o*n3, n3, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv14 = RotConvBN(half_kernel_height,   o*n3, n3, relu=True, bn=True, squeeze=rotconv_squeeze)
        
        self.upsample3 = nn.ConvTranspose2d(o*n3,o*n2, kernel_size=2, stride=2)
        self.conv15 = RotConvBN(half_kernel_height, 2*o*n2, n2, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv16 = RotConvBN(half_kernel_height,   o*n2, n2, relu=True, bn=True, squeeze=rotconv_squeeze)
        
        self.upsample4 = nn.ConvTranspose2d(o*n2,o*n1, kernel_size=2, stride=2)
        self.conv17 = RotConvBN(half_kernel_height, 2*o*n1, n1, relu=True, bn=True, squeeze=rotconv_squeeze)
        self.conv18 = RotConvBN(half_kernel_height,   o*n1, n1, relu=True, bn=True, squeeze=rotconv_squeeze)

        # End
        self.final_conv = nn.Conv2d(o*n1, 1, kernel_size=1)
        
        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else lambda x: x

    def forward(self, x):
        if self.principal_direction=='all':
            downSampledX = x.mean(axis=1)
        else:
            downSampledX = x[:,self.principal_direction]
        device = self.final_conv.weight.device
        pDir1 = principal_direction(downSampledX,device=device,
                                    hessian_value_threshold=self.principal_direction_hessian_threshold,
                                    smooth_std=self.principal_direction_smooth)
        downSampledX = F.avg_pool2d(downSampledX, 2)
        pDir2 = principal_direction(downSampledX,device=device,
                                    hessian_value_threshold=self.principal_direction_hessian_threshold,
                                    smooth_std=self.principal_direction_smooth)
        downSampledX = F.avg_pool2d(downSampledX, 2)
        pDir3 = principal_direction(downSampledX,device=device,
                                    hessian_value_threshold=self.principal_direction_hessian_threshold,
                                    smooth_std=self.principal_direction_smooth)
        downSampledX = F.avg_pool2d(downSampledX, 2)
        pDir4 = principal_direction(downSampledX,device=device,
                                    hessian_value_threshold=self.principal_direction_hessian_threshold,
                                    smooth_std=self.principal_direction_smooth)
        downSampledX = F.avg_pool2d(downSampledX, 2)
        pDir5 = principal_direction(downSampledX,device=device,
                                    hessian_value_threshold=self.principal_direction_hessian_threshold,
                                    smooth_std=self.principal_direction_smooth)
        
        # Down
        x1 = self.conv2(self.conv1(x, project=pDir1), project=pDir1)
        
        x2 = self.pool1(x1)
        x2 = self.conv4(self.conv3(x2, project=pDir2), project=pDir2)
        
        x3 = self.pool2(x2)
        x3 = self.conv6(self.conv5(x3, project=pDir3), project=pDir3)
        
        x4 = self.pool3(x3)
        x4 = self.conv8(self.conv7(x4, project=pDir4), project=pDir4)
        
        x = self.pool4(x4)
        x = self.dropout(self.conv9(x, project=pDir5))
        x = self.dropout(self.conv10(x, project=pDir5))
        
        # Up
        x4 = cat_crop(x4, self.upsample1(x))
        x4 = self.conv12(self.conv11(x4, project=pDir4), project=pDir4)
        
        x3 = cat_crop(x3, self.upsample2(x4))
        x3 = self.conv14(self.conv13(x3, project=pDir3), project=pDir3)
        
        x2 = cat_crop(x2, self.upsample3(x3))
        x2 = self.conv16(self.conv15(x2, project=pDir2), project=pDir2)
        
        x1 = cat_crop(x1, self.upsample4(x2))
        x1 = self.conv18(self.conv17(x1, project=pDir1), project=pDir1)
        
        # End
        return self.final_conv(x1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, depth=2, kernel=3, dilation=1):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.kernel = kernel

        self.model = []

        if in_channels != out_channels:
            conv = ConvBN(1, in_channels, out_channels)
            self.model += [conv]

        for i in range(depth):
            conv = ConvBN(kernel, out_channels, out_channels, dilation=dilation)
            self.model += [conv]
        seq = []
        for m in self.model:
            seq += m.model
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

    def __getattr__(self, attr):
        if attr.startswith('conv') or attr.startswith('bn') or attr.startswith('relu'):
            if attr.startswith('bn'):
                i = int(attr[len('bn'):])
                attr = 'bn'
            elif attr.startswith('relu'):
                i = int(attr[len('relu'):])
                attr = 'relu'
            elif attr.startswith('convbn'):
                i = int(attr[len('convbn'):])
                attr = 'convbn'
            else:
                i = int(attr[len('conv'):])
                attr = 'conv'

            if self.in_channels != self.out_channels:
                convbn = self.model[i]
            else:
                if i == 0:
                    raise AttributeError('Invalid attribute "conv0".')
                convbn = self.model[i - 1]

            return {'convbn': convbn,
                    'conv': convbn.conv,
                    'bn': convbn.bn,
                    'relu': convbn.relu}[convbn]
        return super(ConvBlock, self).__getattr__(attr)


class ConvBN(nn.Module):
    def __init__(self, kernel, n_in, n_out=None, stride=1, relu=True, padding=0, dilation=1, bn=False):
        super(ConvBN, self).__init__()

        self._bn = bn
        if n_out is None:
            n_out = n_in

        model = [nn.Conv2d(n_in, n_out, kernel_size=kernel, stride=stride, padding=padding, bias=False,
                           dilation=dilation)]

        if bn:
            model += [nn.BatchNorm2d(n_out)]
            if relu:
                model += [nn.ReLU()]
        elif relu:
            model += [nn.SELU()]

        self.model = nn.Sequential(*model)

        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # if bn:
        # nn.init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.model(x)

    @property
    def conv(self):
        return self.model[0]

    @property
    def bn(self):
        if self._bn:
            return self.model[1]
        return None

    @property
    def relu(self):
        return self.model[2 if self._bn else 1]


def clip_pad_center(tensor, shape, pad_mode='constant', pad_value=0):
    s = tensor.shape[-2:]

    y0 = (s[0] - shape[-2]) // 2
    y1 = 0
    if y0 < 0:
        y1 = -y0
        y0 = 0

    x0 = (s[1] - shape[-1]) // 2
    x1 = 0
    if x0 < 0:
        x1 = -x0
        x0 = 0
    tensor = tensor[..., y0:y0 + shape[-2], x0:x0 + shape[-1]]
    if x1 or y1:
        tensor = F.pad(tensor, (y1, y1, x1, x1), mode=pad_mode, value=pad_value)
    return tensor


def clip_center(tensor, shape):
    s = tensor.shape[-2:]
    y0 = (s[0] - shape[-2]) // 2
    x0 = (s[1] - shape[-1]) // 2
    return tensor[..., y0:y0 + shape[-2], x0:x0 + shape[-1]]


def neg_pad(t, pad):
    even = pad // 2
    odd = pad - even
    return t[..., even:-odd, even:-odd]


def cat_crop(x1, x2):
    return torch.cat((clip_center(x1, x2.shape), x2), 1)