import junno.datasets as D
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
        
def compute_mask(raw):
    from skimage.filters import median
    from skimage.morphology import disk, remove_small_objects
    raw = median(raw.mean(0), disk(4))
    return remove_small_objects(raw>.05, 250)
    
# DATASETS
def messidor_clinic():
    raw = D.images('/home/gaby/These/Data/Fundus/MESSIDOR1500/0-images/', 'raw')
    av = D.images('/home/gaby/These/Data/Fundus/MESSIDOR1500/Annotation Clinique/Vessels/Vessels - Uncertain/', 'av')
    return D.join(raw, av)

def messidor_mask(name):
    return cv2.imread('/home/gaby/These/Data/Fundus/MESSIDOR1500/mask.png')[...,0]!=0

def messidor_av_legacy():
    raw = D.images('/home/gaby/These/Data/Fundus/MESSIDOR/0-images/', 'raw')
    av = D.images('/home/gaby/These/Data/Fundus/MESSIDOR/2-label_AV/', 'av')
    return D.join(raw, av)

def messidor_val():
    raw = D.images('/home/gaby/These/Data/Fundus/MESSIDOR1500/0-images/', 'raw')
    av = D.images('/home/gaby/These/Data/Fundus/MESSIDOR1500/2-label_AV/', 'av')
    return D.join(raw, av)
    

def drive_train(reshape=True):
    kwargs = dict(reshape=565, keep_proportion='crop') if reshape else {}
    raw = D.images('/home/gaby/These/Data/Fundus/DRIVE/train/0-images/', 'raw', **kwargs)
    av = D.images('/home/gaby/These/Data/Fundus/DRIVE/train/2-label_AV/', 'av', **kwargs)
    return D.join(raw, av)

def drive_test(reshape=True):
    kwargs = dict(reshape=565, keep_proportion='crop') if reshape else {}
    raw = D.images('/home/gaby/These/Data/Fundus/DRIVE/test/original/0-images/', 'raw', **kwargs)
    av = D.images('/home/gaby/These/Data/Fundus/DRIVE/test/original/2-label_AV/', 'av', **kwargs)
    ct = D.images('/home/gaby/These/Data/Fundus/DRIVE/test/original/2-label_AV_CT/', 'ct', **kwargs)
    ct2px = D.images('/home/gaby/These/Data/Fundus/DRIVE/test/original/2-label_AV_CT2px/', 'ct2px', **kwargs)
    return D.join(raw, av, ct, ct2px)
        
def hrf():
    kwargs = dict()#reshape=2000, keep_proportion='pad')
    raw = D.images('/home/gaby/These/Data/Fundus/HRF/0-images/', 'raw', **kwargs)
    av = D.images('/home/gaby/These/Data/Fundus/HRF/2-label_AV/', 'av', **kwargs)
    mask = D.images('/home/gaby/These/Data/Fundus/HRF/1-mask/', 'mask', **kwargs).apply('mask', lambda x: x[0]>.7)
    return D.join(raw, av, mask)


def DRIVE(train=True, preprocessing=True):
    if train:
        d = drive_train()
    else:
        d = drive_test().apply('ct', _av2label, format=[{1:'red', 2:'blue', 3:'white'}]).apply('ct2px', _av2label, format=[{1:'red', 2:'blue', 3:'white'}])
    d = d.apply('av', _av2label, format=[{1:'red', 2:'blue', 3:'white'}])
    if preprocessing:
        d = d.apply({'pre':'raw'}, preprocess, keep_parent=True, format='same')
    return d

def MESSIDOR(preprocessing=True, train=True, av=True):
    if av:
        d = messidor_val()
        if train:
            d = d.subset(stop=40)
        else:
            d = d.subset(start=40)
        d = d.apply('av', _av2label, format=[{1:'red', 2:'blue', 3:'white'}])
    else:
        if train:
            d = messidor_clinic()
            d = d.apply('av', _v2label, format=[{1:'white'}])
        else:
            d = messidor_val()
            d = d.apply('av', _av2label, format=[{1:'red', 2:'blue', 3:'white'}])
    if preprocessing:
        d = d.apply({'pre':'raw'}, preprocess, keep_parent=True, format='same')
    d = d.apply('mask', messidor_mask, keep_parent=True)
    return d

def HRF(train=True, preprocessing=True):
    d = hrf()
    d = d.apply('av', _av2label, format=[{1:'red', 2:'blue', 3:'white'}])
    
    if preprocessing:
        d = d.apply({'pre':'raw'}, preprocess, keep_parent=True, format='same')
    return d

# UTILS

def _av2label(av):
    label = np.zeros(av.shape[-2:], np.int)
    label[av[0]>0.2] = 2
    label[av[2]>0.2] = 1
    label[av[1]>0.2] = 3
    return label

def _v2label(av):
    label = np.zeros(av.shape[-2:], np.bool)
    label[av.sum(0)!=0] = 1
    return label

def fundus_preprocessing(x):
    k = np.max(x.shape)//20*2+1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k//2+1,k//2+1))
    
    mask_org = (x[0,:,:]>10/255.).astype(np.uint8)
    mask =  cv2.erode(mask_org, np.ones((15,15), np.uint8))
    
    mask = np.expand_dims(mask, 0).astype(np.uint8)
    mask_org = np.expand_dims(mask_org, 2)
    x_cv = (x*255).transpose((1,2,0)).astype(np.uint8)
    dilation = cv2.dilate(x_cv,kernel,iterations = 1)
    dilation = dilation.astype(np.float).transpose((2,0,1))/255.
    fundus = preprocess(dilation*(1-mask)+mask*x)*mask
    return fundus

def preprocess(img):
    sigma = np.max(img.shape)/60
    blur = np.stack([gaussian_filter(img[0], sigma, truncate=6.5),
                     gaussian_filter(img[1], sigma, truncate=6.5),
                     gaussian_filter(img[2], sigma, truncate=6.5) ])
    return (img - blur - .0022501)/.02771

def low_pass_thresh(mask):
    mask= cv2.blur((mask*255).astype(np.uint8), (128,128))
    return mask > mask.max()-1

def draw_mask(raw):
    w, h = raw.shape[-2:]
    r = int(np.ceil(min(w,h)*.45))
    img = np.zeros((w,h), dtype=np.uint8)
    cv2.circle(img, (w//2, h//2), r, 255, -1)
    return img>0
