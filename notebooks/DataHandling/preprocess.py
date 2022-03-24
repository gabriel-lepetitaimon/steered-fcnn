from skimage.morphology import skeletonize
import torch
import numpy as np
import scipy.stats as st


def gkern(n=21, sigma=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-sigma, sigma, n+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


_G_xy_cached = {}


def G_xy(std):
    if std in _G_xy_cached:
        return _G_xy_cached[std]
    n = std*12+1
    x = np.linspace(-std*3,std*3,n)
    y, x = np.meshgrid(x,x)
    G = (gkern(n,std)+1e-6)  / (np.sqrt(x*x+y*y)+1e-8)
    x *= G
    y *= G
    G = np.stack((x,y))
    _G_xy_cached[std] = G
    return G


def compute_field_torch(skeleton, std=None):
    if std is None:
        std = int(np.ceil(max(skeleton.shape[-2:]) / 20)) # std = ceil(max(h,w)/20)
    with torch.no_grad():
        skeleton = torch.from_numpy(skeleton).cuda().double()
        G = torch.from_numpy(G_xy(std))[:,None,:,:].cuda()
        f = torch.conv2d(skeleton[:,None,:,:], G)
        f = f[:,0]+1j*f[:,1]
        return f.cpu().numpy()


def compute_skeleton_field(binary_map, std=None):
    if std is None:
        std = int(np.ceil(max(binary_map.shape[-2:]) / 20)) # std = ceil(max(h,w)/20)
    
    G = G_xy(std)
    n = G.shape[0]
    half = n//2
    
    skeleton = skeletonize(binary_map)
    sk_grad = np.zeros(shape=(2,)+skeleton.shape, dtype=np.float32)
    h, w = skeleton.shape
    for i,j in np.argwhere(skeleton):
        i1=max(half-i,0)
        i2=min(n,h+half-i)
        h0=i2-i1
        i0=i+i1-half
        
        j1=max(half-j,0)
        j2=min(n,w+half-j)
        w0=j2-j1
        j0 = j+j1-half
        sk_grad[:, i0:i0+h0, j0:j0+w0] += G[:,i1:i2,j1:j2]

    return sk_grad


############################################################################################################
#       ---  VESSELS PREPROCESSING  ---

def skeletonize_av(av):
    return skeletonize(av==1)+skeletonize(av==2)
