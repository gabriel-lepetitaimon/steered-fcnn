import os.path as P
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import time

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

    file = h5py.File(dataset_file, mode='r')
    trainD = LogIdleTimeDataLoader(TrainDataset('train', file=file,
                                   factor=cfg.training['training-dataset-factor'],
                                   data_augmentation_cfg=cfg['data-augmentation']),
                                   pin_memory=True, shuffle=True,
                                   batch_size=batch_size, patch_shape=patch_shape,
                                   num_workers=cfg.training['num-worker'])
    validD = LogIdleTimeDataLoader(BaseDataset('val', file=file,  patch_shape=patch_shape),
                                   pin_memory=True, num_workers=4, batch_size=4)
    testD = {'test': LogIdleTimeDataLoader(BaseDataset('test', file=file,  patch_shape=patch_shape),
                                           pin_memory=True, num_workers=4, batch_size=4)}
    return trainD, validD, testD


class LogIdleTimeDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(LogIdleTimeDataLoader, self).__init__(*args, **kwargs)
        self._time_logs = []

    def avg_idle(self):
        sum_iter = sum(_['iter'] for _ in self._time_logs)
        avg_idle = sum(_['idle'] for _ in self._time_logs) / sum_iter
        avg_wait = sum(_['wait'] for _ in self._time_logs) / sum_iter
        return avg_idle - avg_wait

    def last_idle(self):
        logs = self._time_logs[-1]
        avg_idle = logs['idle'] / logs['iter']
        avg_wait = logs['wait'] / logs['iter']
        return avg_idle - avg_wait

    def __iter__(self):
        iter = super(LogIdleTimeDataLoader, self).__iter__()
        t_idle = 0
        t_ini_wait = 0

        logs = {'wait': 0., 'idle': 0., 'iter': 0, 'ini': 0}
        self._time_logs.append(logs)

        while True:
            t0 = time.time()
            it = next(iter)
            t_wait = time.time() - t0

            if not t_ini_wait:
                t_ini_wait = t_wait
                logs['ini'] = t_wait
            else:
                t_idle -= t_ini_wait
                if t_idle > t_wait:
                    logs['idle'] += t_idle
            logs['iter'] += 1
            logs['wait'] += t_wait

            t0 = time.time()
            yield it
            t_idle = time.time() - t0


class BaseDataset(Dataset):
    def __init__(self, dataset, file, patch_shape=None, mode='segment'):
        super(BaseDataset, self).__init__()
        self.x = file.get(f'{dataset}/x')
        self.y = file.get(f'{dataset}/y')
        self.data_fields = dict(images='x', labels='y')
        self._data_length = len(self.x)
        self.patch_shape = patch_shape
        self.DA = DataAugment()
        self.mode=mode
        self.factor = 4
        self.transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Efficient net

    def __len__(self):
        return self._data_length*self.factor

    @property
    def geo_aug(self):
        compiled = getattr(self, '__geo_aug_compiled', None)
        if compiled:
            return compiled

        compiled = self.DA.compile(self.data_fields, to_torch=True)
        self.__geo_aug_compiled = compiled
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
        data['mask'] = y > 0
        if self.mode == 'segment':
            data['y'] = y > 1
        else:
            h, w = data['y'].shape
            data['y'] = y[h//2, w//2] > 1
        if self.transforms:
            data['x'] = self.transforms(data['x'])
        return data


class TrainDataset(BaseDataset):
    def __init__(self, dataset, file, factor=1, patch_shape=None, data_augmentation_cfg=None):
        super(TrainDataset, self).__init__(dataset=dataset, file=file, patch_shape=patch_shape)

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


def crop_pad(img, size, center=None):
    h, w = size
    if center is None:
        y, x = h//2, w//2
    else:
        y, x = center
    half_x = size[1] // 2
    odd_x = size[1] % 2
    half_y = size[0] // 2
    odd_y = size[0] % 2

    y0 = int(max(0, half_y - y))
    y1 = int(max(0, y - half_y))
    h = int(min(h, y + half_y + odd_y) - y1)

    x0 = int(max(0, half_x - x))
    x1 = int(max(0, x - half_x))
    w = int(min(w, x + half_x + odd_x) - x1)

    r = np.zeros_like(img, shape=size+img.shape[2:])
    r[y0:y0+h, x0:x0+w] = img[y1:y1+h, x1:x1+w]
    return r