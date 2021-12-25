import torch
from torch import nn
from ..utils import cat_crop, pyramid_pool2d, normalize_vector
from ..steered_conv import SteeredConvBN, SteeredConvTranspose2d, DEFAULT_STEERABLE_BASE
from ..steered_conv.steerable_filters import cos_sin_ka_stack
from .backbones import UNet


class SteeredUNet(UNet):
    def __init__(self, n_in, n_out, nfeatures=6, depth=2, nscale=5, padding='same',
                 p_dropout=0, batchnorm=True, downsampling='maxpooling', upsampling='conv',
                 base=DEFAULT_STEERABLE_BASE, attention_base=None, attention_mode='shared', normalize_steer=False):
        if base is None:
            base = DEFAULT_STEERABLE_BASE
        self.base = base
        self.attention_base = attention_base

        super(SteeredUNet, self).__init__(n_in, n_out, nfeatures=nfeatures, kernel=kernel, depth=depth,
                                          nscale=nscale, padding=padding, p_dropout=p_dropout, batchnorm=batchnorm,
                                          downsampling=downsampling, upsampling=upsampling,
                                          attention_mode=attention_mode, normalize_steer=normalize_steer)

    def setup_convbn(self, n_in, n_out):
        return SteeredConvBN(n_in, n_out, steerable_base=self.base, attention_base=self.attention_base,
                             attention_mode=self.attention_mode, normalize_steer_vec=self.normalize_steer,
                             relu=True, bn=self.batchnorm, padding=self.padding)

    def setup_convtranspose(self, n_in, n_out):
        return SteeredConvTranspose2d(n_in, n_out, stride=2)

    def forward(self, x, alpha=None, rho=None):
        N = 5
        if alpha is None:
            if self.attention_base is None:
                raise ValueError('If no attention base is specified, a steering angle alpha should be provided.')
            else:
                alpha_pyramid = [None]*N
                rho_pyramid = [None]*N
        else:
            with torch.no_grad():
                k_max = self.base.k_max

                rho = 1
                if alpha.dim() == 3:
                    cos_sin_kalpha = cos_sin_ka_stack(torch.cos(alpha), torch.sin(alpha), k=k_max)
                elif alpha.dim() == 4 and alpha.shape[1] == 2:
                    alpha = alpha.transpose(0, 1)
                    alpha, rho = normalize_vector(alpha)
                    if self.normalize_steer is True:
                        rho = 1
                    elif self.normalize_steer == 'tanh':
                        rho = torch.tanh(rho)
                    cos_sin_kalpha = cos_sin_ka_stack(alpha[0], alpha[1], k=k_max)
                else:
                    raise ValueError(f'alpha shape should be either [b, h, w] or [b, 2, h, w] '
                                     f'but provided tensor shape is {alpha.shape}.')
                cos_sin_kalpha = cos_sin_kalpha.unsqueeze(3)

                alpha_pyramid = pyramid_pool2d(cos_sin_kalpha, n=N)
                rho_pyramid = [rho]*N if not isinstance(rho, torch.Tensor) else pyramid_pool2d(rho, n=N)

        xscale = []
        for i, conv_stack in enumerate(self.down_conv[:-1]):
            x = self.reduce_stack(conv_stack, x, alpha=alpha_pyramid[i], rho=rho_pyramid[i])
            xscale += [self.dropout(x)]
            x = self.downsample(x)

        x = self.reduce_stack(self.down_conv[-1], x, alpha=alpha_pyramid[-1], rho=rho_pyramid[-1])
        x = self.dropout(x)

        for i, conv_stack in enumerate(self.up_conv):
            x = cat_crop(xscale.pop(), self.upsample(x))
            x = self.reduce_stack(conv_stack, x, alpha=alpha_pyramid[-i], rho=rho_pyramid[-i])

        return self.final_conv(x)


class SteeredHemelingNet(nn.Module):
    def __init__(self, n_in, n_out=1, nfeatures_base=6, depth=2, base=None, attention=None,
                 p_dropout=0, padding='same', batchnorm=True, upsample='conv'):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.upsample = upsample

        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16

        # Down
        self.conv1 = nn.ModuleList(
            [SteeredConvBN(n_in, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n1, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.ModuleList(
            [SteeredConvBN(n1, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n2, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.ModuleList(
            [SteeredConvBN(n2, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n3, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.ModuleList(
            [SteeredConvBN(n3, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n4, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.ModuleList(
            [SteeredConvBN(n4, n5, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n5, n5, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        # Up
        if upsample == 'nearest':
            self.upsample1 = nn.Sequential(nn.Conv2d(n5, n4, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.ModuleList(
            [SteeredConvBN(2 * n4, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n4, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        if upsample == 'nearest':
            self.upsample2 = nn.Sequential(nn.Conv2d(n4, n3, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample2 = nn.ConvTranspose2d(n4, n3, kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.ModuleList(
            [SteeredConvBN(2 * n3, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n3, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        if upsample == 'nearest':
            self.upsample3 = nn.Sequential(nn.Conv2d(n3, n2, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.ModuleList(
            [SteeredConvBN(2 * n2, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n2, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        if upsample == 'nearest':
            self.upsample4 = nn.Sequential(nn.Conv2d(n2, n1, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.ModuleList(
            [SteeredConvBN(2 * n1, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n1, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=(1, 1))

        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else identity

    @property
    def attention(self):
        return self.conv1[0].conv.attention_base

    @property
    def base(self):
        return self.conv1[0].conv.steerable_base

    def forward(self, x, alpha=None, **kwargs):
        """
        Args:
            x: The input tensor.
            alpha: The angle by which the network is steered. (If None then alpha=0.)
                    To enhance the efficiency of the steering computation, α should not be provided as an angle but as a
                    vertical and horizontal projection: cos(α) and sin(α).
                    Hence this parameter should be a 4D tensor: alpha[b, 0, h, w]=cos(α) and alpha[b, 1, h, w]=sin(α).
                    (Alpha can be broadcasted along b, h or w, if these dimensions are of length 1.)
                    Default: None

        Shape:
            input: (b, n_in, h, w)
            alpha: (b, 2, ~h, ~w)     (b, h and w are broadcastable)
            return: (b, n_out, ~h, ~w)

        Returns: The prediction of the network (without the sigmoid).

        """
        from functools import reduce

        N = 5
        if alpha is None:
            if self.attention is None:
                raise NotImplementedError()
            else:
                alpha_pyramid = [None]*N
                rho_pyramid = [None]*N
        else:
            with torch.no_grad():
                k_max = self.base.k_max

                rho = 1
                if alpha.dim() == 3:
                    cos_sin_kalpha = cos_sin_ka_stack(torch.cos(alpha), torch.sin(alpha), k=k_max)
                elif alpha.dim() == 4 and alpha.shape[1] == 2:
                    alpha = alpha.transpose(0, 1)
                    alpha, rho = normalize_vector(alpha)
                    cos_sin_kalpha = cos_sin_ka_stack(alpha[0], alpha[1], k=k_max)
                else:
                    raise ValueError(f'alpha shape should be either [b, h, w] or [b, 2, h, w] '
                                     f'but provided tensor shape is {alpha.shape}.')
                cos_sin_kalpha = cos_sin_kalpha.unsqueeze(3)

                alpha_pyramid = pyramid_pool2d(cos_sin_kalpha, n=N)
                rho_pyramid = [rho]*N if not isinstance(rho, torch.Tensor) else pyramid_pool2d(rho, n=N)

        # Down
        x1 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[0], rho=rho_pyramid[0]), self.conv1, x)

        x2 = self.pool1(x1)
        x2 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[1], rho=rho_pyramid[1]), self.conv2, x2)

        x3 = self.pool2(x2)
        x3 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[2], rho=rho_pyramid[2]), self.conv3, x3)

        x4 = self.pool3(x3)
        x4 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[3], rho=rho_pyramid[3]), self.conv4, x4)

        x5 = self.pool4(x4)
        x5 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[4], rho=rho_pyramid[4]), self.conv5, x5)
        x5 = self.dropout(x5)

        # Up
        x4 = cat_crop(x4, self.upsample1(x5))
        del x5
        x4 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[3], rho=rho_pyramid[3]), self.conv6, x4)

        x3 = cat_crop(x3, self.upsample2(x4))
        del x4
        x3 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[2], rho=rho_pyramid[2]), self.conv7, x3)

        x2 = cat_crop(x2, self.upsample3(x3))
        del x3
        x2 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[1], rho=rho_pyramid[1]), self.conv8, x2)

        x1 = cat_crop(x1, self.upsample4(x2))
        del x2
        x1 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[0], rho=rho_pyramid[0]), self.conv9, x1)

        # End
        return self.final_conv(x1)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p

        
def identity(x):
    return x