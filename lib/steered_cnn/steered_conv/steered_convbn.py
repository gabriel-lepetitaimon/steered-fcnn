from torch import nn
from .steered_conv import SteeredConv2d


class SteeredConvBN(nn.Module):
    def __init__(self, kernel, n_in, n_out=None, steerable_base=None, stride=1, padding='same', dilation=1,
                 attention_base=None, attention_mode='feature', normalize_steer_vec=None,
                 groups=1, bn=False, relu=True):
        super().__init__()

        self._bn = bn
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        self.conv = SteeredConv2d(kernel, n_in, n_out, steerable_base=steerable_base, stride=stride, groups=groups,
                                  padding=padding, bias=not bn, dilation=dilation, attention_base=attention_base,
                                  attention_mode=attention_mode, normalize_steer_vec=normalize_steer_vec,
                                  nonlinearity='relu')
        bn_relu = []
        if bn:
            bn_relu += [nn.BatchNorm2d(self.n_out)]
            if relu:
                bn_relu += [nn.ReLU()]
        elif relu:
            bn_relu += [nn.SELU()]

        self.bn_relu = nn.Sequential(*bn_relu)

    def forward(self, x, alpha=None, rho=None):
        x = self.conv(x, alpha=alpha, rho=rho)
        return self.bn_relu(x)

    @property
    def bn(self):
        if self._bn:
            return self.model[0]
        return None

    @property
    def relu(self):
        return self.model[1 if self._bn else 0]
