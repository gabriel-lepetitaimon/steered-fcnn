import os.path as P
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ..config import default_config
from .data_augment import DataAugment


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


def load_dataset(cfg=None):
    if cfg is None:
        cfg = default_config()
    batch_size = cfg['hyper-parameters']['batch-size']
    steered = cfg.get('model.steered.steering', False)
    if steered == 'attention':
        steered = False
    train_dataset = cfg.training['training-dataset']
    data_path = cfg.training.get('dataset-path', 'default')
    if data_path == 'default':
        data_path = DEFAULT_DATA_PATH
    dataset_file = P.join(data_path, cfg.training['dataset-file'])
    trainD = DataLoader(TrainDataset('train/'+train_dataset, file=dataset_file,
                                     factor=cfg.training['training-dataset-factor'],
                                     steered=steered,
                                     data_augmentation_cfg=cfg['data-augmentation']),
                        pin_memory=True, shuffle=True,
                        batch_size=batch_size,
                        num_workers=cfg.training['num-worker']
                        )
    validD = DataLoader(TestDataset('val/'+train_dataset, file=dataset_file, steered=steered),
                        pin_memory=True, num_workers=6, batch_size=6)
    testD = {_: DataLoader(TestDataset('test/'+_, file=dataset_file, steered=steered),
                           pin_memory=True, num_workers=6, batch_size=6)
             for _ in ('MESSIDOR', 'HRF', 'DRIVE')}
    return trainD, validD, testD


class TrainDataset(Dataset):
    def __init__(self, dataset, file, factor=1, steered=True, data_augmentation_cfg=None):
        super(TrainDataset, self).__init__()

        if data_augmentation_cfg is None:
            data_augmentation_cfg = default_config()['data-augmentation']

        with h5py.File(file, 'r') as DATA:
            self.x = DATA.get(f'{dataset}/data')
            self.y = DATA.get(f'{dataset}/av')
            self.mask = DATA.get(f'{dataset}/mask')
            data_fields = dict(images='x', labels='y,mask')

            if steered:
                if steered is True:
                    steered = 'vec-norm'
                if steered == 'vec':
                    self.steer = DATA.get(f'{dataset}/principal-vec-norm')
                    data_fields['vectors'] = 'alpha'
                elif steered == 'vec-norm':
                    self.steer = DATA.get(f'{dataset}/principal-vec')
                    data_fields['vectors'] = 'alpha'
                elif steered == 'angle':
                    data_fields['angles'] = 'alpha'
                    self.steer = DATA.get(f'{dataset}/principal-angle')
                elif steered == 'all':
                    self.angle = DATA.get(f'{dataset}/principal-angle')
                    self.vec = DATA.get(f'{dataset}/principal-vec-norm')  # Through sigmoid
                    self.vec_norm = DATA.get(f'{dataset}/principal-vec')
                    data_fields['angles'] = 'angle'
                    data_fields['vectors'] = 'vec,vec_norm,angle_xy'
                else:
                    raise ValueError(
                        'steered should be one of "vec", "vec-norm", "angle" or "all".'
                        f'(Provided: "{steered}")')
        
        DA = DataAugment().flip()
        if data_augmentation_cfg['rotation']:
            DA.rotate()
        if data_augmentation_cfg['elastic']:
            DA.elastic_distortion(alpha=data_augmentation_cfg['elastic-transform']['alpha'],
                                  sigma=data_augmentation_cfg['elastic-transform']['sigma'],
                                  alpha_affine=data_augmentation_cfg['elastic-transform']['alpha-affine']
                                  )

        self.geo_aug = DA.compile(**data_fields, to_torch=True)
        self.DA = DA
        self.steered = steered
        self.factor = factor
        self._data_length = len(self.x)

    def __len__(self):
        return self._data_length * self.factor

    def __getitem__(self, i):
        i = i % self._data_length
        x = self.x[i].transpose(1, 2, 0)
        if self.steered:
            if isinstance(self.steered, str) and self.steered != 'all':
                alpha = self.steer[i]
                if self.steered != 'angle':
                    alpha = alpha.transpose(1, 2, 0)
                    if self.steered == 'vec-norm':
                        alpha = alpha / (np.linalg.norm(alpha, axis=2, keepdims=True) + 1e-8)
                return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i], alpha=alpha)
            else:
                angle = self.angle[i]
                vec = self.vec[i].transpose(1, 2, 0)
                vec_norm = self.vec_norm[i].transpose(1, 2, 0)
                vec_norm = vec_norm / (np.linalg.norm(vec_norm, axis=2, keepdims=True) + 1e-8)
                angle_xy = np.stack([np.cos(angle), np.sin(angle)], axis=2)
                return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i], angle=angle, vec=vec, vec_norm=vec_norm, angle_xy=angle_xy)
        else:
            return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i])


class TestDataset(Dataset):
    def __init__(self, dataset, file, steered=True):
        super(TestDataset, self).__init__()
        DATA = h5py.File(file, 'r')

        self.x = DATA.get(f'{dataset}/data')
        self.y = DATA.get(f'{dataset}/av')
        self.mask = DATA.get(f'{dataset}/mask')
        data_fields = dict(images='x', labels='y,mask')

        if steered:
            if steered is True:
                steered = 'vec-norm'
            if steered == 'vec':
                self.steer = DATA.get(f'{dataset}/principal-vec-norm')
                data_fields['vectors'] = 'alpha'
            elif steered == 'vec-norm':
                self.steer = DATA.get(f'{dataset}/principal-vec')
                data_fields['vectors'] = 'alpha'
            elif steered == 'angle':
                data_fields['angles'] = 'alpha'
                self.steer = DATA.get(f'{dataset}/principal-angle')
            elif steered == 'all':
                self.angle = DATA.get(f'{dataset}/principal-angle')
                self.vec = DATA.get(f'{dataset}/principal-vec-norm')  # Through sigmoid
                self.vec_norm = DATA.get(f'{dataset}/principal-vec')
                data_fields['angles'] = 'angle'
                data_fields['vectors'] = 'vec,vec_norm'
            else:
                raise ValueError(
                    'steered should be one of "vec", "vec-norm", "angle" or "all".'
                    f'(Provided: "{steered}")')

        self.geo_aug = DataAugment().compile(**data_fields, to_torch=True)
        
        self.steered = steered
        self._data_length = len(self.x)

    def __len__(self):
        return self._data_length

    def __getitem__(self, i):
        x = self.x[i].transpose(1, 2, 0)
        if self.steered:
            if isinstance(self.steered, str) and self.steered != 'all':
                alpha = self.steer[i]
                if self.steered != 'angle':
                    alpha = alpha.transpose(1, 2, 0)
                    if self.steered == 'vec-norm':
                         alpha = alpha / (np.linalg.norm(alpha, axis=2, keepdims=True) + 1e-8)
                return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i], alpha=alpha)
            else:
                angle = self.angle[i]
                vec = self.vec[i].transpose(1, 2, 0)
                vec_norm = self.vec_norm[i].transpose(1, 2, 0)
                vec_norm = vec_norm / (np.linalg.norm(vec_norm, axis=2, keepdims=True) + 1e-8)
                return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i], angle=angle, vec=vec, vec_norm=vec_norm)
        else:
            return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i])
