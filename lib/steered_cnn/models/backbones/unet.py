import torch
from torch import nn
from ...utils import cat_crop
from ...utils.convbn import ConvBN
from .model import Model


class UNet(Model):
    def __init__(self, n_in, n_out=1, nfeatures_base=64, kernel=3, depth=2, nscale=5, padding='same',
                 p_dropout=0, batchnorm=True, downsampling='maxpooling', upsampling='conv'):
        """
        :param n_in:
        :param n_out:
        :param nfeatures_base:
        :param depth:
        :param nscale:
        :param padding:
        :param p_dropout:
        :param batchnorm:
        :param downsampling:
            - maxpooling: Maxpooling Layer.
            - averagepooling: Average Pooling.
            - conv: Stride on the last convolution.
        :param upsampling:
            - conv: "Deconvolution with stride"
            - bilinear: "Bilinear upsampling"
            - nearest: "Nearest upsampling"
        """
        super().__init__(n_in=n_in, n_out=n_out, nfeatures_base=nfeatures_base, depth=depth, nscale=nscale,
                         kernel=kernel, padding=padding, p_dropout=p_dropout, batchnorm=batchnorm,
                         downsampling=downsampling, upsampling=upsampling)

        # Down
        self.down_conv = []
        for i in range(nscale):
            nf_prev = n_in if i == 0 else (nfeatures_base * (2**(i-1)))
            nf_scale = nfeatures_base * (2**i)
            conv_stack = [self.setup_convbn(nf_prev, nf_scale)]
            conv_stack += [self.setup_convbn(nf_scale, nf_scale) for _ in range(depth-1)]
            if downsampling == 'conv':
                conv_stack[-1].stride = 2
            self.down_conv += [conv_stack]

        self.up_conv = []
        for i in reversed(range(nscale-1)):
            nf_scale = nfeatures_base * 3 * (2**i)
            nf_next = nfeatures_base * (2**i)
            conv_stack = [self.setup_convbn(nf_scale, nf_scale) for _ in range(depth-1)]
            if upsampling == 'conv':
                conv_stack += [self.setup_convtranspose(nf_scale, nf_next)]
            else:
                conv_stack += [self.setup_convstack(nf_scale, nf_next)]
            self.up_conv += [conv_stack]

        # End
        self.final_conv = nn.Conv2d(nfeatures_base, n_out, kernel_size=(1, 1))

        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else identity
        if downsampling == 'maxpooling':
            self.downsample = torch.nn.MaxPool2d(2)
        elif downsampling == 'averagepooling':
            self.downsample = torch.nn.AvgPool2d(2)
        elif downsampling == 'conv':
            self.downsample = identity
        else:
            raise ValueError(f'downsampling must be one of: "maxpooling", "averagepooling", "conv". '
                             f'Provided: {downsampling}.')
        if upsampling == 'bilinear':
            self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        elif upsampling == 'nearest':
            self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        elif upsampling == 'conv':
            self.upsample = identity
        else:
            raise ValueError(f'upsampling must be one of: "bilinear", "nearest", "conv". '
                             f'Provided: {upsampling}.')

    def setup_convbn(self, n_in, n_out):
        return ConvBN(self.kernel, n_in, n_out, relu=True, bn=self.batchnorm, padding=self.padding)

    def setup_convtranspose(self, n_in, n_out):
        return torch.nn.ConvTranspose2d(n_in, n_out, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
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
        xscale = []
        for conv_stack in self.down_conv[:-1]:
            x = self.reduce_stack(conv_stack, x)
            xscale += [self.dropout(x)]
            x = self.downsample(x)

        x = self.reduce_stack(self.down_conv[-1], x)
        x = self.dropout(x)

        for conv_stack in self.up_conv:
            x = cat_crop(xscale.pop(), self.upsample(x))
            x = self.reduce_stack(conv_stack, x)

        return self.final_conv(x)

    def reduce_stack(self, conv_stack, x, **kwargs):
        from functools import reduce

        def conv(X, conv_mod):
            return conv_mod(X, **kwargs)
        return reduce(conv, conv_stack, x)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p


def identity(x):
    return x
