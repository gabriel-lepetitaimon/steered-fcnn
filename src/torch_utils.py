import torch.nn.functional as F
import torch


# --- Simple Module ---
class ConvBN(torch.nn.Module):
    def __init__(self, kernel, n_in, n_out=None, stride=1, relu=True, padding=0, dilation=1, bn=False):
        super(ConvBN, self).__init__()

        self._bn = bn
        if n_out is None:
            n_out = n_in

        if padding == 'auto':
            padding = kernel//2, kernel//2

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
def clip_pad_center(tensor, shape, pad_mode='constant', pad_value=0):
    s = tensor.shape

    y0 = (s[-2]-shape[-2])//2
    y1 = 0
    yodd = (shape[-2]-s[-2]) % 2
    if y0 < 0:
        y1 = -y0
        y0 = 0

    x0 = (s[-1]-shape[-1])//2
    x1 = 0
    xodd = (shape[-1]-s[-1]) % 2
    if x0 < 0:
        x1 = -x0
        x0 = 0
    tensor = tensor[..., y0:y0+shape[-2], x0:x0+shape[-1]]
    if x1 or y1:
        tensor = F.pad(tensor, (y1-yodd, y1, x1-xodd, x1), mode=pad_mode, value=pad_value)
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

    t1 = t1[..., i1:i1+h, j1:j1+w]
    t2 = t2[..., i2:i2+h, j2:j2+w]
    return t1, t2


def pad_tensors(t1, t2, pad_mode='constant', pad_value=0):
    if t1.shape[-2:] == t2.shape[-2:]:
        return t1, t2

    def half_odd(v):
        return v//2, v % 2

    h1, w1 = t1.shape[-2:]
    h2, w2 = t2.shape[-2:]

    dh = h1-h2
    dh2 = max(dh, 0)
    dh1, dh1_odd = half_odd(dh2-dh)
    dh2, dh2_odd = half_odd(dh2)

    dw = w1-w2
    dw2 = max(dw, 0)
    dw1, dw1_odd = half_odd(dw2-dw)
    dw2, dw2_odd = half_odd(dw2)

    if dw1+dw1_odd or dh1+dh1_odd:
        t1 = F.pad(t1, (dh1, dh1+dh1_odd, dw1, dw1+dw1_odd), mode=pad_mode, value=pad_value)
    if dw2+dw2_odd or dh2+dh2_odd:
        t2 = F.pad(t2, (dh2, dh2+dh2_odd, dw2, dw2+dw2_odd), mode=pad_mode, value=pad_value)
    return t1, t2


def neg_pad(t, pad):
    even = pad // 2
    odd = pad - even
    return t[..., even:-odd, even:-odd]


def cat_crop(x1, x2):
    return torch.cat(clip_tensors(x1, x2), 1)


def get_padding(padding, shape):
    if padding == 'auto':
        hW, wW = shape[-2:]
        padding = (hW//2, wW//2)
    elif isinstance(padding, int):
        padding = (padding, padding)
    return padding


