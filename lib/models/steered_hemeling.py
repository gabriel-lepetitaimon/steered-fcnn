import torch
from torch import nn
import torch.nn.functional as F
from ..utils import cat_crop
from ..steered_conv import SteeredConvBN, SteerableKernelBase
from ..steered_conv.steerable_filters import cos_sin_ka_stack


class SteeredHemelingNet(nn.Module):

    def __init__(self, n_in, n_out=1, nfeatures_base=6, depth=2, base=None,
                 p_dropout=0, padding=0,
                 static_principal_direction=False):
        super(SteeredHemelingNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.static_principal_direction = static_principal_direction

        if base is None:
            base = SteerableKernelBase.create_from_rk(4, max_k=5)
        elif isinstance(base, (int, dict)):
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
               for _ in range(depth - 1)])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.ModuleList(
            [SteeredConvBN(n1, n2, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n2, n2, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.ModuleList(
            [SteeredConvBN(n2, n3, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n3, n3, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.ModuleList(
            [SteeredConvBN(n3, n4, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n4, n4, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.ModuleList(
            [SteeredConvBN(n4, n5, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n5, n5, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])

        # Up
        self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.ModuleList(
            [SteeredConvBN(2 * n4, n4, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n4, n4, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])

        self.upsample2 = nn.ConvTranspose2d(n4, n3, kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.ModuleList(
            [SteeredConvBN(2 * n3, n3, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n3, n3, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])

        self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.ModuleList(
            [SteeredConvBN(2 * n2, n2, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n2, n2, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])

        self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.ModuleList(
            [SteeredConvBN(2 * n1, n1, relu=True, bn=True, padding=padding, steerable_base=base)]
            + [SteeredConvBN(n1, n1, relu=True, bn=True, padding=padding, steerable_base=base)
               for _ in range(depth - 1)])

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=(1, 1))

        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else lambda x: x

    def forward(self, x, alpha=None):
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
        if alpha is None:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                k_max = self.base.k_max

                if alpha.dim == 3:
                    cos_sin_kalpha = cos_sin_ka_stack(torch.cos(alpha), torch.sin(alpha), k=k_max)
                elif alpha.dim == 4 and alpha.shape[1] == 2:
                    cos_sin_kalpha = cos_sin_ka_stack(alpha[:, 0], alpha[:, 1], k=k_max)
                else:
                    raise ValueError(f'alpha shape should be either [b, h, w] or [b, 2, h, w] '
                                     f'but provided tensor shape is {alpha.shape}.')

                _, k_max, b, h, w = alpha.shape

                alpha1 = cos_sin_kalpha.reshape((2 * k_max, b, h, w))
                alpha2 = F.avg_pool2d(alpha1, 2)
                alpha3 = F.avg_pool2d(alpha2, 2)
                alpha4 = F.avg_pool2d(alpha3, 2)
                alpha5 = F.avg_pool2d(alpha4, 2)

                alpha1 = tuple(alpha1.reshape(2, k_max, b, 1, *alpha1.shape[-2:]))
                alpha2 = tuple(alpha2.reshape(2, k_max, b, 1, *alpha2.shape[-2:]))
                alpha3 = tuple(alpha3.reshape(2, k_max, b, 1, *alpha3.shape[-2:]))
                alpha4 = tuple(alpha4.reshape(2, k_max, b, 1, *alpha4.shape[-2:]))
                alpha5 = tuple(alpha5.reshape(2, k_max, b, 1, *alpha5.shape[-2:]))

        # Down
        x1 = reduce(lambda X, conv: conv(X, alpha=alpha1), self.conv1, x)

        x2 = self.pool1(x1)
        x2 = reduce(lambda X, conv: conv(X, alpha=alpha2), self.conv2, x2)

        x3 = self.pool2(x2)
        x3 = reduce(lambda X, conv: conv(X, alpha=alpha3), self.conv3, x3)

        x4 = self.pool3(x3)
        x4 = reduce(lambda X, conv: conv(X, alpha=alpha4), self.conv4, x4)

        x5 = self.pool4(x4)
        x5 = reduce(lambda X, conv: conv(X, alpha=alpha5), self.conv5, x5)
        x5 = self.dropout(x5)

        # Up
        x4 = cat_crop(x4, self.upsample1(x5))
        del x5
        x4 = reduce(lambda X, conv: conv(X, alpha=alpha4), self.conv6, x4)

        x3 = cat_crop(x3, self.upsample2(x4))
        del x4
        x3 = reduce(lambda X, conv: conv(X, alpha=alpha3), self.conv7, x3)

        x2 = cat_crop(x2, self.upsample3(x3))
        del x3
        x2 = reduce(lambda X, conv: conv(X, alpha=alpha2), self.conv8, x2)

        x1 = cat_crop(x1, self.upsample4(x2))
        del x2
        x1 = reduce(lambda X, conv: conv(X, alpha=alpha1), self.conv9, x1)

        # End
        return self.final_conv(x1)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p
