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
    data_path = cfg.training.get('dataset-path', 'default')
    if data_path == 'default':
        data_path = DEFAULT_DATA_PATH
    dataset_file = P.join(data_path, cfg.training['dataset-file'])

    file = h5py.File(dataset_file, mode='r')
    trainD = DataLoader(TrainDataset('train', file=file,
                                     factor=cfg.training['training-dataset-factor'],
                                     data_augmentation_cfg=cfg['data-augmentation']),
                        pin_memory=True, shuffle=True,
                        batch_size=batch_size,
                        num_workers=cfg.training['num-worker']
                        )
    validD = DataLoader(TestDataset('val', file=file),
                        pin_memory=True, num_workers=4, batch_size=4)
    testD = {'test': DataLoader(TestDataset('test', file=file),
                           pin_memory=True, num_workers=4, batch_size=4)}
    return trainD, validD, testD


class TrainDataset(Dataset):
    def __init__(self, dataset, file, factor=1, data_augmentation_cfg=None):
        super(TrainDataset, self).__init__()

        if data_augmentation_cfg is None:
            data_augmentation_cfg = default_config()['data-augmentation']

        self.x = file.get(f'{dataset}/x')  # [:]
        self.y = file.get(f'{dataset}/y')  # [:]
        data_fields = dict(images='x', labels='y')

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
        self.factor = factor
        self._data_length = len(self.x)

    def __len__(self):
        return self._data_length * self.factor

    def __getitem__(self, i):
        i = i % self._data_length
        x = self.x[i].transpose(1, 2, 0)
        data = self.geo_aug(x=x, y=self.y[i,0])
        y = data['y']
        data['mask'] = y > 0
        data['y'] = y > 1
        return data


class TestDataset(Dataset):
    def __init__(self, dataset, file):
        super(TestDataset, self).__init__()

        print(list(file.keys()))
        
        self.x = file.get(f'{dataset}/x')
        self.y = file.get(f'{dataset}/y')
        data_fields = dict(images='x', labels='y')

        self.geo_aug = DataAugment().compile(**data_fields, to_torch=True)

        self._data_length = len(self.x)

    def __len__(self):
        return self._data_length

    def __getitem__(self, i):
        x = self.x[i].transpose(1, 2, 0)
        data = self.geo_aug(x=x, y=self.y[i,0])
        y = data['y']
        data['mask'] = y > 0
        data['y'] = y > 1
        return data
