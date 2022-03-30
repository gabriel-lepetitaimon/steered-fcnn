import os.path
import os.path as P
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import time
import pandas as pd
import cv2

from ..config import default_config
from .data_augment import DataAugment


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


def load_dataset(cfg=None):
    if cfg is None:
        cfg = default_config()
    batch_size = cfg['hyper-parameters']['batch-size']
    data_path = cfg.training.get('dataset-path', 'default')
    if data_path == 'default':
        data_path = DEFAULT_DATA_PATH
    dataset_file = P.join(data_path, cfg.training['dataset-file'])
    
    patch_shape = cfg.training.get('patch', None)
    if isinstance(patch_shape, (float, int)):
        h = int(patch_shape)
        patch_shape = h,h
    elif isinstance(patch_shape, (tuple,list)):
        patch_shape = tuple(int(_) for _ in patch_shape)
        
    mode = cfg.training.get('mode', 'segment')

    if mode=='segment':
        file = h5py.File(dataset_file, mode='r')
        trainD = TrainSegmentDataset('train', file=file, patch_shape=patch_shape,
                                     factor=cfg.training['training-dataset-factor'],
                                     data_augmentation_cfg=cfg['data-augmentation'])
        validD = SegmentDataset('val', file=file, patch_shape=patch_shape)
        testD = {'test': SegmentDataset('test', file=file, patch_shape=patch_shape)}
        train_args = {'shuffle': True}
    else:
        raw_path = P.join(data_path, cfg.training['raw-path'])
        trainD = TrainClassifyDataset('train', file=dataset_file, patch_shape=patch_shape, raw_path=raw_path,
                                      data_augmentation_cfg=cfg['data-augmentation'])
        validD = ClassifyDataset('val', file=dataset_file, patch_shape=patch_shape, raw_path=raw_path,)
        testD = {'test': ClassifyDataset('test', file=dataset_file, patch_shape=patch_shape, raw_path=raw_path,)}
        
        ratio = cfg.training.get('ratio-ma-art', 7)
        repeat = cfg.training.get('art-repeat-factor', 1)
        train_args = {
            'sampler': torch.utils.data.WeightedRandomSampler([ratio/trainD.length_ma]*trainD.length_ma + [1/(trainD.length_art*repeat)]*trainD.length_art*repeat,
                                                              num_samples=int(min(trainD.length_art*repeat*(1+1/ratio), trainD.length_ma*(1+ratio))))}

    trainD = LogIdleTimeDataLoader(trainD, batch_size=batch_size, num_workers=cfg.training['num-worker'], **train_args)
    validD = LogIdleTimeDataLoader(validD, num_workers=cfg.training['num-worker'], batch_size=batch_size)
    testD = {k: LogIdleTimeDataLoader(d, num_workers=cfg.training['num-worker'], batch_size=batch_size, shuffle=True) for k, d in testD.items()}
    return trainD, validD, testD


def worker_init_fn(worker_id):
    # set seed
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32-1))


class LogIdleTimeDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(LogIdleTimeDataLoader, self).__init__(*args, **kwargs, worker_init_fn=worker_init_fn, pin_memory=True)
        self._time_logs = []

    def avg_idle(self):
        sum_iter = sum(_['iter'] for _ in self._time_logs)
        avg_idle = sum(_['idle'] for _ in self._time_logs) / sum_iter
        avg_wait = sum(_['wait'] for _ in self._time_logs) / sum_iter
        return avg_idle - avg_wait

    def avg_total(self):
        return sum(_['total'] for _ in self._time_logs) / sum(_['iter'] for _ in self._time_logs)

    def last_idle(self):
        logs = self._time_logs[-1]
        avg_idle = logs['idle'] / logs['iter']
        avg_wait = logs['wait'] / logs['iter']
        return avg_idle - avg_wait

    def last_total(self):
        logs = self._time_logs[-1]
        return logs['total'] / logs['iter']

    def __iter__(self):
        iter = super(LogIdleTimeDataLoader, self).__iter__()
        t_idle = 0
        t_ini_wait = 0

        logs = {'wait': 0., 'idle': 0., 'total': 0., 'iter': 0, 'ini': 0}
        self._time_logs.append(logs)

        while True:
            t0 = time.time()
            try:
                it = next(iter)
            except StopIteration:
                return
            t_wait = time.time() - t0

            if not t_ini_wait:
                t_ini_wait = t_wait
                logs['ini'] = t_wait
            else:
                t_idle -= t_ini_wait
                if t_idle > t_wait:
                    logs['idle'] += t_idle
                logs['total'] += t_idle + t_wait + t_ini_wait
            logs['iter'] += 1
            logs['wait'] += t_wait

            t0 = time.time()
            yield it
            t_idle = time.time() - t0


class SegmentDataset(Dataset):
    def __init__(self, dataset, file, patch_shape=None):
        super(SegmentDataset, self).__init__()
        self.x = file.get(f'{dataset}/x')
        self.y = file.get(f'{dataset}/y')
        self.data_fields = dict(images='x', labels='y')
        self._data_length = len(self.x)
        self.patch_shape = patch_shape
        self.DA = DataAugment()
        self.factor = 4
        self.transforms = None #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Efficient net

    def __len__(self):
        return self._data_length*self.factor

    @property
    def geo_aug(self):
        compiled = getattr(self, '_geo_aug', None)
        if compiled:
            return compiled
        
        compiled = self.DA.compile(**self.data_fields, to_torch=True)
        self._geo_aug = compiled
        return compiled

    def __getitem__(self, i):
        i = i % self._data_length

        x = self.x[i]
        y = self.y[i, 0]

        if self.patch_shape:
            P_artefact = 0.5
            p = np.random.binomial(1, P_artefact)+1
            center = np.where(y == p)
            center_id = np.random.randint(len(center[0]))
            center = tuple(_[center_id] for _ in center)

            large_patch_shape = tuple(_*np.sqrt(2) for _ in self.patch_shape)
            x = crop_pad(x, center=center, size=large_patch_shape)
            y = crop_pad(y, center=center, size=large_patch_shape)

        x = x.transpose((1, 2, 0))
        data = self.geo_aug(x=x, y=y)

        if self.patch_shape:
            data['x'] = crop_pad(data['x'], size=self.patch_shape)
            y = crop_pad(data['y'], size=self.patch_shape)
        else:
            y = data['y']
        data['y'] = y > 1
        data['mask'] = y > 0

        if self.transforms:
            data['x'] = self.transforms(data['x'])

        return data


class TrainSegmentDataset(SegmentDataset):
    def __init__(self, dataset, file, factor=1, patch_shape=None, data_augmentation_cfg=None):
        super(TrainSegmentDataset, self).__init__(dataset=dataset, file=file, patch_shape=patch_shape)

        if data_augmentation_cfg is None:
            data_augmentation_cfg = default_config()['data-augmentation']

        DA = DataAugment().flip()
        if data_augmentation_cfg['rotation']:
            DA.rotate()
        if data_augmentation_cfg['elastic']:
            DA.elastic_distortion(alpha=data_augmentation_cfg['elastic-transform']['alpha'],
                                  sigma=data_augmentation_cfg['elastic-transform']['sigma'],
                                  alpha_affine=data_augmentation_cfg['elastic-transform']['alpha-affine']
                                  )
        if 'hue' in data_augmentation_cfg or 'saturation' in data_augmentation_cfg or 'value' in data_augmentation_cfg:
            DA.hsv(hue=data_augmentation_cfg.get('hue', None),
                   saturation=data_augmentation_cfg.get('saturation', None),
                   value=data_augmentation_cfg.get('value', None))
        self.DA = DA
        self.factor = factor


class ClassifyDataset(Dataset):
    def __init__(self, dataset, file, raw_path, patch_shape):
        super(ClassifyDataset, self).__init__()
        self.raw_path = raw_path

        art_centers = pd.read_excel(file, dataset+'-art')
        art_centers['label'] = 1
        self.art_centers = art_centers.to_numpy()

        ma_centers = pd.read_excel(file, dataset+'-ma')
        ma_centers['label'] = 0
        self.ma_centers = ma_centers.to_numpy()

        self.data_fields = dict(images='x')
        self.patch_shape = patch_shape
        self.DA = DataAugment()
        self._geo_aug = None
        self.transforms = None #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Efficient net

    def __len__(self):
        return self.length_ma+self.length_art
    
    @property
    def length_ma(self):
        return len(self.ma_centers)
    
    @property
    def length_art(self):
        return len(self.art_centers)

    @property
    def geo_aug(self):
        if self._geo_aug:
            return self._geo_aug

        geo_aug = self.DA.compile(**self.data_fields, to_torch=True)
        self._geo_aug = geo_aug
        return geo_aug

    def __getitem__(self, i):
        len_ma = self.length_ma
        if i < len_ma:
            _, filename, center_y, center_x, y = self.ma_centers[i]
        else:
            _, filename, center_y, center_x, y = self.art_centers[(i-len_ma) % self.length_art]
        center = center_y, center_x

        large_patch_shape = tuple(int(round(_*2)) for _ in self.patch_shape)

        x = cv2.imread(os.path.join(self.raw_path, filename))
        x = crop_pad(x, center=center, size=large_patch_shape)
        x = x.astype(np.float32)/255
        data = self.geo_aug(x=x)
        data['x'] = crop_pad(data['x'], size=self.patch_shape)
        if self.transforms:
            data['x'] = self.transforms(data['x'])
        data['y'] = y
        return data


class TrainClassifyDataset(ClassifyDataset):
    def __init__(self, dataset, file, raw_path, patch_shape, resample_factor=1, data_augmentation_cfg=None):
        super(TrainClassifyDataset, self).__init__(dataset=dataset, file=file, raw_path=raw_path,
                                                   patch_shape=patch_shape)

        self.resample_factor = resample_factor

        if data_augmentation_cfg is None:
            data_augmentation_cfg = default_config()['data-augmentation']

        DA = DataAugment().flip()
        if data_augmentation_cfg['rotation']:
            DA.rotate()
        if data_augmentation_cfg['elastic']:
            DA.elastic_distortion(alpha=data_augmentation_cfg['elastic-transform']['alpha'],
                                  sigma=data_augmentation_cfg['elastic-transform']['sigma'],
                                  alpha_affine=data_augmentation_cfg['elastic-transform']['alpha-affine']
                                  )
        if 'hue' in data_augmentation_cfg or 'saturation' in data_augmentation_cfg or 'value' in data_augmentation_cfg:
            DA.hsv(hue=data_augmentation_cfg.get('hue', None),
                   saturation=data_augmentation_cfg.get('saturation', None),
                   value=data_augmentation_cfg.get('value', None))
        self.DA = DA

    def __len__(self):
        return len(self.ma_centers)+len(self.art_centers)*self.resample_factor


def crop_pad(img, size, center=None):
    if torch.is_tensor(img):
        H, W = img.shape[-2:]
    else:
        H, W = img.shape[:2]
    h, w = size
    if center is None:
        y, x = H//2, W//2
    else:
        y, x = center
    half_w = w // 2
    odd_w = w % 2
    half_h = h // 2
    odd_h = h % 2

    y0 = int(max(0, half_h - y))
    y1 = int(max(0, y - half_h))
    h = int(min(h, H-y1) - y0)

    x0 = int(max(0, half_w - w))
    x1 = int(max(0, x - half_w))
    w = int(min(w, W-x1) - x0)
    
    if torch.is_tensor(img):
        r = torch.zeros(tuple(img.shape[:-2])+size, dtype=img.dtype, device=img.device)
        r[..., y0:y0+h, x0:x0+w] = img[..., y1:y1+h, x1:x1+w]
    else:
        r = np.zeros_like(img, shape=size+img.shape[2:])
        r[y0:y0+h, x0:x0+w] = img[y1:y1+h, x1:x1+w]
    
    return r
