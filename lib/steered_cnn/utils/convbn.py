import torch


class ConvBN(torch.nn.Module):
    def __init__(self, kernel, n_in, n_out=None, stride=1, relu=True, padding=0, dilation=1, bn=False):
        super().__init__()

        self._bn = bn
        if n_out is None:
            n_out = n_in
        if isinstance(kernel, int):
            kernel = kernel, kernel
        padding = compute_padding(padding, kernel)

        model = [torch.nn.Conv2d(n_in, n_out, kernel_size=kernel, stride=stride, padding=padding, bias=False,
                                 dilation=dilation)]

        if bn:
            model += [torch.nn.BatchNorm2d(n_out)]
            if relu:
                model += [torch.nn.ReLU()]
        elif relu:
            model += [torch.nn.SELU()]

        self.model = torch.nn.Sequential(*model)

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


# --- Utils function ---
def compute_padding(padding, shape):
    if padding == 'same' or padding == 'auto':
        hW, wW = shape[-2:]
        padding = (hW//2, wW//2)
    elif padding == 'true' or padding == 'valid':
        padding = (0, 0)
    elif padding == 'full':
        hW, wW = shape[-2:]
        padding = (hW-hW % 2, wW-wW % 2)
    elif isinstance(padding, int):
        padding = (padding, padding)
    return padding


def compute_conv_outputs_dim(input_shape, weight_shape, padding=0, stride=1, dilation=1):
    h, w = input_shape[-2:]
    n, m = weight_shape[-2:]

    if not isinstance(padding, tuple):
        padding = compute_padding(padding, weight_shape)
    if isinstance(stride, int):
        stride = stride, stride
    if isinstance(dilation, int):
        dilation = dilation, dilation
    h = int((h+2*padding[0]-dilation[0]*(n-1)-1)/stride[0] + 1)
    w = int((w+2*padding[1]-dilation[1]*(m-1)-1)/stride[1] + 1)
    return h, w


def pyramid_pool2d(input_tensor, n):
    import torch.nn.functional as F
    s = input_tensor.shape
    h, w = s[-2:]

    if h != 1 and w != 1:
        if len(s) > 4:
            t = input_tensor.reshape(torch.prod(s[:-2]), h, w)
        else:
            t = input_tensor
        pyramid = [input_tensor]
        for _ in range(n):
            t = F.avg_pool2d(t, 2)
            pyramid += [t.reshape(s[:-2]+(h, w))]
    else:
        return [input_tensor] * n

    return pyramid
