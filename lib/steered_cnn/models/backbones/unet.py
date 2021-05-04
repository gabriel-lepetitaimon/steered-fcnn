import torch
from torch import nn
from ...utils import cat_crop
from ...utils.convbn import ConvBN
from .model import Model


class UNet(Model):
    def __init__(self, n_in, n_out=1, nfeatures_base=6, depth=2, nscale=5, padding='same',
                 p_dropout=0, batchnorm=True, downsample='maxpooling', upsample='conv'):
        """

        :param n_in:
        :param n_out:
        :param nfeatures_base:
        :param depth:
        :param nscale:
        :param padding:
        :param p_dropout:
        :param batchnorm:
        :param downsample:
            - maxpooling: Maxpooling Layer.
            - averagepooling: Average Pooling.
            - conv: Stride on the last convolution.
        :param upsample:
            - conv: "Deconvolution with stride"
            - bilinear: "Bilinear upsampling"
            - nearest: "Nearest upsampling"
        """
        super().__init__(n_in=n_in, n_out=n_out, nfeatures_base=nfeatures_base, depth=depth, nscale=nscale,
                         padding=padding, p_dropout=p_dropout, batchnorm=batchnorm,
                         downsample=downsample, upsample=upsample)

        # Down
        self.down_conv = []
        for i in range(nscale):
            nf_prev = n_in if i == 0 else (nfeatures_base * (2**(i-1)))
            nf_scale = nfeatures_base * (2**i)
            conv_stack = [self.setup_convbn(nf_prev if _ == 0 else nf_scale, nf_scale) for _ in range(depth)]
            if downsample == 'conv':
                conv_stack[-1].stride = 2
            self.down_conv += [conv_stack]

        self.up_conv = []
        for i in reversed(range(nscale)):
            nf_scale = nfeatures_base * (2**i)
            nf_next = nfeatures_base * (2**(i-1))
            conv_stack = [self.setup_convbn(nf_scale, nf_scale) for _ in range(depth-1)]
            if upsample == 'conv':
                conv_stack += [self.setup_convtranspose(nf_scale, nf_next)]
            else:
                conv_stack += [self.setup_convstack(nf_scale, nf_next)]
            self.up_conv += [conv_stack]

        # End
        self.final_conv = nn.Conv2d(nfeatures_base, n_out, kernel_size=(1, 1))

        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else identity

    def setup_convbn(self, n_in, n_out):
        return ConvBN(n_in, n_out, relu=True, bn=self.batchnorm, padding=self.padding)

    def setup_convtranspose(self, n_in, n_out):
        return torch.nn.ConvTranspose2d(n_in, n_out, kernel_size=(2, 2), stride=(2, 2))

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

        xscale = []
        for conv_stack in  
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