import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
plt.rcParams["figure.figsize"] = (18, 5)
from skimage.transform import rotate as imrotate
from scipy.ndimage import gaussian_filter, convolve1d, convolve
import cv2

def polar_space(size, center=None):
    if isinstance(size, int):
        size = (size,size)
    if center is None:
        center = tuple(_/2 for _ in size)
        
    y = np.linspace(-center[0], size[0]-center[0], size[0])
    x = np.linspace(-center[1], size[1]-center[1], size[1])
    y,x = np.meshgrid(y,x)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def radial_steerable_filter(size, k, r, std=.5):
    rho, phi = polar_space(size)
    G = np.exp(-(rho-r)**2/(2 * std**2))
    PHI = np.exp(1j*k*phi)
    return G*PHI

def plot_field(vy, vx, vr=None, mask=None, clip_norm=1, background=None):
    if vr is None:
        vr = np.sqrt(np.square(vx)+np.square(vy))
        vr = np.clip(vr/vr.max(),0.0000001,clip_norm)
        vy = vy/vr
        vx = vx/vr
    else:
        vr = np.clip(vr/vr.max(),0.0000001,clip_norm)
    vy = vy * np.log(1+np.abs(vr))**.5 *2
    vx = vx * np.log(1+np.abs(vr))**.5 *2
    
    fig, ax = plt.subplots()
    if background is not None:
        if mask is not None:
            back = np.zeros_like(background)
            mask = mask!=0
            back[mask] = background[mask]
            background = back
        ax.imshow(background, cmap='gray', vmin=0)
    if mask is not None:
        r = vr[mask!=0]
        v = vy[mask!=0]
        u = vx[mask!=0]
        y,x = mask.nonzero()
        ax.quiver( x, y, u, -v, r, cmap='copper')
    else:
        ax.quiver( vx, -vy, vr, cmap='copper')
    fig.set_size_inches(18,18)
    fig.show()
    
def plot_filter(F, axis=True, spd=False):
    h, w = F.shape
    v = max(F.max(), -F.min())
    
    if spd is True:
        spd = 16
    
    if spd:
        fig, (ax_filt, ax_spd) = plt.subplots(1,2);
    else:
        fig, ax_filt = plt.subplots();

    # --- PLOT FILTER ---
    im = ax_filt.imshow(F, interpolation='none', vmin=-v, vmax=v, aspect='equal', cmap='RdGy')
    if axis:
            # Major ticks
        ax_filt.set_xticks(np.arange(0, w, 1))
        ax_filt.set_yticks(np.arange(0, h, 1))

        # Labels for major ticks
        ax_filt.set_xticklabels(np.arange(1, w+1, 1))
        ax_filt.set_yticklabels(np.arange(1, h+1, 1))
    else:
        ax_filt.set_xticklabels([])
        ax_filt.set_yticklabels([])
        ax_filt.set_xticks([])
        ax_filt.set_yticks([])
    
    # Minor ticks
    ax_filt.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax_filt.set_yticks(np.arange(-.5, h, 1), minor=True)

    # Gridlines based on minor ticks
    ax_filt.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    # --- PLOT SPD ---
    if spd:
        polar_spectral_power(F, plot=ax_spd, theta=spd)
    fig.tight_layout(w_pad=-3)
    fig.show()
    
DEFAULT_ANGLE = np.arange(10,360,10)
def rotate(arr, angles=None):
    shape = arr.shape
    if angles is None:
        angles = DEFAULT_ANGLE
    if isinstance(angles, int):
        angles = np.linspace(0, 360, angles, endpoint=False)[1:]
    arr = arr.reshape((-1,)+arr.shape[-2:]).transpose((1,2,0))
    arr = np.stack([arr]+[imrotate(arr, -a) for a in angles])
    return arr.transpose((0,3,1,2)).reshape((len(angles)+1,)+shape)


def unrotate(arr:'θ.hw', angles=None)->'θ.hw':
    shape = arr.shape
    if angles is None:
        angles = DEFAULT_ANGLE
    if isinstance(angles, int):
        angles = np.linspace(0,360, angles)[1:]
    arr = arr.reshape((arr.shape[0],-1)+arr.shape[-2:]).transpose((0,2,3,1))
    arr = np.stack([arr[0]]+
                    [imrotate(ar, ang) for ar, ang in zip(arr[1:],angles)])
    return arr.transpose((0,3,1,2)).reshape(shape)


def polar_spectral_power(arr: '.hw', theta=8, plot=False, split=False):
    arr = rotate(arr, theta)
    return spectral_power(arr, plot=plot, split=split)
    

def spectral_power(arr: 'θ.hw', plot=False, split=False):
    from scipy import fft
    spe = fft(arr, axis=0)
    spe = abs(spe)**2
    if split:
        spe = spe.reshape(spe.shape[:2]+(-1,)).sum(axis=-1)
    else:
        spe = spe.reshape(spe.shape[:1]+(-1,)).sum(axis=1)
    if plot:
        fig = None
        scale = False
        if isinstance(plot, str):
            scale = plot
            plot = True
        if plot is True:
            fig, plot = plt.subplots()
      
        N = len(spe)//2+1
        
        if split:
            W = 0.8
            w = W/spe.shape[1]
            for i in range(spe.shape[1]):
                y = spe[:len(spe)//2+1, i]
                if split == 'normed':
                    y = y/y.sum()
                else:
                    y = y / spe[:N].sum(axis=-1).mean()
                x = np.arange(len(y))
                plot.bar(x+w/2-W/2+i*w, y, width=w, bottom=0.001, zorder=10)
        else:
            y = spe[:N] / spe[:len(spe)//2+1].sum()
            x = np.arange(len(y))
            plot.bar(x, y, width=.8, bottom=0.001, zorder=10, color='gray')

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.spines['left'].set_visible(False)

        #plot.set_xlabel('Polar harmonic')
        plot.set_xticks(np.arange(0, N, 1))
        xlabels = ['Equivariant','$2\pi$', '$\pi$'][:min(3,N)]
        xlabels += ['$\dfrac{2\pi}{%i}$'%_ if _%2 else '$\dfrac{\pi}{%i}$'%(_//2)
                    for _ in range(3,N)]
        plot.set_xticklabels(xlabels)
        
        plot.set_ylabel('Polar Spectral Power Density')
        plot.set_ylim([0.001,1])
        plot.set_yticks([.25,.5,.75, 1])
        plot.set_yticklabels(['25%','50%','75%', '100%'])
        plot.yaxis.grid()
        if scale:
            plot.set_yscale(scale)
        plot.grid(which='minor', color='#bbbbbb', linestyle='-', linewidth=1, zorder=1)
        
            
        if fig is not None:
            fig.show()
    return spe

def field_unrotate(arrs, angles=None, reproject=True):
    if angles is None:
        angles = DEFAULT_ANGLE
    y,x = arrs
    y = unrotate(y, angles)
    x = unrotate(x, angles)
    
    z = x+1j*y
    angle_offset = np.concatenate([[0], DEFAULT_ANGLE])
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

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def field_reconstruction_error(arrs, angles=None, plot=True, imshow=False, mask=None, reproject=False, angle_mod=1, norm_threshold=0):
    if angles is None:
        angles = DEFAULT_ANGLE
    arrs = field_unrotate(arrs, angles=angles, reproject=reproject)
    
    if not reproject:
        theta, r = arrs
        if norm_threshold:
            mask = mask[np.newaxis]*(r>norm_threshold)
        t_diff = compute_error(theta, xaxis=angles, plot=plot, imshow=imshow, mask=mask, angle_mod=angle_mod, title="angle")
        r_diff = compute_error(r, xaxis=angles, plot=plot, imshow=imshow, mask=mask, title="r")
        return t_diff[0], r_diff[0]
    else:
        p, o = arrs
        p_diff = reconstruction_error(p, angles=angles, plot=plot, imshow=imshow, mask=mask, title="principal")
        o_diff = reconstruction_error(o, angles=angles, plot=plot, imshow=imshow, mask=mask, title="orthogonal")
    return p_diff[0], o_diff[0]


def reconstruction_error(arr, angles=None, plot=True, imshow=False, mask=None, title=""):
    if angles is None:
        angles = DEFAULT_ANGLE
    arr = unrotate(arr, angles=angles)
    return compute_error(arr, plot=plot, imshow=imshow, mask=mask, xaxis=angles, title=title)
    

def compute_error(arr, plot=True, imshow=False, mask=None, xaxis=None, angle_mod=False, title=""):
    l = arr.shape[0]-1
    diff_mean = np.zeros((l,))
    diff_std = np.zeros((l,))
    if imshow:
        diff_img = np.zeros(arr.shape[-2:])
        
    base = arr[0]
    for i, current in enumerate(arr[1:]):
        diff = current-base
        if angle_mod:
            if angle_mod is True:
                angle_mod = 1
            diff = simplify_angle(diff, angle_mod)
        diff = np.abs(diff)
            
        if mask is None:
            diff_mean[i] = diff.mean()
            diff_std[i] = diff.std()
            if imshow:
                diff_img += diff.mean(axis=0)
        else:
            if(mask.ndim==4):
                current_mask = mask[i]
            elif(mask.ndim==3):
                current_mask = mask[i,np.newaxis]
            elif(mask.ndim==2):
                current_mask = mask[np.newaxis]
            else:
                raise ValueError()
            if current_mask.shape[0]!=diff.shape[0]:
                current_mask = np.repeat(current_mask, diff.shape[0], axis=0)
            try:
                diff_mean[i], diff_std[i] = weighted_avg_and_std(diff, current_mask)
            except ZeroDivisionError:
                diff_mean[i], diff_std[i] = 0, 0
            except TypeError as e:
                print(diff.shape, current_mask.shape)
                raise e
            if imshow:
                diff_img += (diff*current_mask).mean(axis=0)
    
    if xaxis is None:
            xaxis = np.arange(l)
    if plot:
        fig = None
        if plot is True:
            fig, plot = plt.subplots()
        plot.bar(xaxis, diff_mean+diff_std, width=2, color='cyan')
        plot.bar(xaxis, diff_mean, width=2, color='blue')
        plot.set_title('Max {} diff={:.2e} at {}$'.format(title, diff_mean.max(), xaxis[np.argmax(diff_mean)]))
        if fig is not None:
            fig.show()

    if imshow:
        diff_img /= l
        fig = None
        if imshow is True:
            fig, imshow = plt.subplots()
        imshow.imshow(diff_img)
        if title:
            imshow.set_title(title)
        if fig is not None:
            fig.show()
    return diff_mean, diff_std


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