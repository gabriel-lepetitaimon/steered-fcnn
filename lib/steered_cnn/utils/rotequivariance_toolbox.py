import numpy as np


def polar_space(size, center=None):
    if not isinstance(size, tuple):
        size = (size, size)
    if center is None:
        center = tuple(_ / 2 for _ in size)

    y = np.linspace(-center[0], size[0] - center[0], size[0])
    x = np.linspace(-center[1], size[1] - center[1], size[1])
    y, x = np.meshgrid(y, x)
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def spectral_power(arr: 'θ.hw', plot=False, split=False, sort=True):
    from scipy import fft
    import matplotlib.pyplot as plt

    spe = fft(arr, axis=0)
    spe = abs(spe) ** 2
    if split:
        spe = spe.reshape(spe.shape[:2] + (-1,)).sum(axis=-1)
    else:
        spe = spe.reshape(spe.shape[:1] + (-1,)).sum(axis=1)
    if plot:
        fig = None
        scale = False
        if isinstance(plot, str):
            scale = plot
            plot = True
        if plot is True:
            fig, plot = plt.subplots()

        N = spe.shape[0] // 2 + 1

        if split:
            W = 0.8
            w = W / spe.shape[1]

            spe = spe[:N]
            if split == 'normed':
                spe = spe / spe.sum(axis=tuple(_ for _ in range(spe.ndim) if _ != 1))[None, :]
            else:
                spe = spe / spe.sum(axis=-1).mean(axis=tuple(_ for _ in range(spe.ndim)
                                                             if _ not in (1, spe.ndim-1)))[None, :]
            if sort:
                idx = spe[0].argsort()
                spe = spe[:, idx[::-1]]
            for i in range(spe.shape[1]):
                y = spe[:, i]
                x = np.arange(len(y))
                plot.bar(x + w / 2 - W / 2 + i * w, y, width=w, bottom=0.001, zorder=10)
        else:
            y = spe[:N] / spe[:N].sum()
            x = np.arange(len(y))
            plot.bar(x, y, width=.8, bottom=0.001, zorder=10, color='gray')

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.spines['left'].set_visible(False)

        plot.set_xticks(np.arange(0, N, 1))
        xlabels = ['Equivariant', '$2\pi$', '$\pi$'][:min(3, N)]
        xlabels += ['$\dfrac{2\pi}{%i}$' % k for k in range(3, N)]
        plot.set_xticklabels(xlabels)

        plot.set_ylabel('Polar Spectral Power Density')
        plot.set_ylim([0.001, 1])
        plot.set_yticks([.25, .5, .75, 1])
        plot.set_yticklabels(['25%', '50%', '75%', '100%'])
        plot.yaxis.grid()
        if scale:
            plot.set_yscale(scale)
        plot.grid(which='minor', color='#bbbbbb', linestyle='-', linewidth=1, zorder=1)

        if fig is not None:
            fig.show()
    return spe


def polar_spectral_power(arr: '.hw', theta=8, plot=False, split=False):
    arr = rotate(arr, theta)
    return spectral_power(arr, plot=plot, split=split)


#####################################################################################
#                       Rotation Equivariance Measures                              #
#####################################################################################
DEFAULT_ROT_ANGLE = np.arange(10, 360, 10)


def rotate(arr, angles=DEFAULT_ROT_ANGLE):
    from skimage.transform import rotate as imrotate

    shape = arr.shape
    if isinstance(angles, int):
        angles = np.linspace(0, 360, angles, endpoint=False)[1:]
    arr = arr.reshape((-1,) + arr.shape[-2:]).transpose((1, 2, 0))
    arr = np.stack([arr] + [imrotate(arr, -a) for a in angles])
    return arr.transpose((0, 3, 1, 2)).reshape((len(angles) + 1,) + shape)


def unrotate(arr: 'θ.hw', angles=DEFAULT_ROT_ANGLE) -> 'θ.hw':
    from skimage.transform import rotate as imrotate

    shape = arr.shape
    if isinstance(angles, int):
        angles = np.linspace(0, 360, angles)[1:]
    arr = arr.reshape((arr.shape[0], -1) + arr.shape[-2:]).transpose((0, 2, 3, 1))
    arr = np.stack([arr[0]] +
                   [imrotate(ar, ang) for ar, ang in zip(arr[1:], angles)])
    return arr.transpose((0, 3, 1, 2)).reshape(shape)


def field_unrotate(arrs, angles=DEFAULT_ROT_ANGLE, reproject=True):
    y,x = arrs
    y = unrotate(y, angles)
    x = unrotate(x, angles)

    z = x+1j*y
    angle_offset = np.concatenate([[0], angles])
    while angle_offset.ndim < z.ndim:
        angle_offset = np.expand_dims(angle_offset, -1)
    theta = (np.angle(z, deg=True) - angle_offset)
    r = np.abs(z)
    if reproject:
        theta *= np.pi/180
        y = r*np.sin(theta)
        x = r*np.cos(theta)
        return y, x
    else:
        return theta, r


def simplify_angle(angles, mod=1, deg=True):
    mod = (360 if deg else 2*np.pi)/mod
    angles = np.mod(angles, mod)
    angles = np.stack([angles, angles-mod])
    a_idx = np.argmin(np.abs(angles), axis=0)
    angles = np.take_along_axis(angles, np.expand_dims(a_idx, axis=0), axis=0).squeeze(0)
    return angles


#####################################################################################
#                           Custom pyplot Scales                                    #
#####################################################################################
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker


class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'sqrt'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        #mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


class SquareScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'sqr'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        #mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareScale.InvertedSquareTransform()

    class InvertedSquareTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**.5

        def inverted(self):
            return SquareScale.SquareTransform()

    def get_transform(self):
        return self.SquareTransform()


mscale.register_scale(SquareRootScale)
mscale.register_scale(SquareScale)
