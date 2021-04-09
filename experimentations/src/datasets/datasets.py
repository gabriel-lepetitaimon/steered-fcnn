import os.path as P
import h5py
from torch.utils.data import Dataset, DataLoader

from ..config import default_config
from .data_augment import DataAugment


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


def load_dataset(cfg=None, data_path=DEFAULT_DATA_PATH):
    if cfg is None:
        cfg = default_config()
    batch_size = cfg['hyper-parameters']['batch-size']
    train_dataset = cfg.training['training-dataset']
    dataset_file = P.join(data_path, cfg.training['dataset-file'])
    trainD = DataLoader(TrainDataset('train/'+train_dataset, file=dataset_file,
                                     factor=cfg.training['training-dataset-factor'],
                                     data_augmentation_cfg=cfg['data-augmentation']),
                        pin_memory=True, shuffle=True,
                        batch_size=batch_size,
                        num_workers=cfg.training['num-worker']
                        )
    validD = DataLoader(TestDataset('val/'+train_dataset, file=dataset_file),
                        pin_memory=True, num_workers=6, batch_size=6)
    testD = {_: DataLoader(TestDataset('test/'+_, file=dataset_file),
                           pin_memory=True, num_workers=6, batch_size=6)
             for _ in ('MESSIDOR', 'HRF', 'DRIVE')}
    return trainD, validD, testD


class TrainDataset(Dataset):
    def __init__(self, dataset, file, factor=1, data_augmentation_cfg=None):
        super(TrainDataset, self).__init__()

        if data_augmentation_cfg is None:
            data_augmentation_cfg = default_config()['data-augmentation']

        DATA = h5py.File(file, 'r')
        self.data = DATA.get(f'{dataset}/data')
        self.av = DATA.get(f'{dataset}/av')
        self.cos_sin_alpha = DATA.get(f'{dataset}/principal-angle')
        self.mask = DATA.get(f'{dataset}/mask')
        
        DA = DataAugment().flip()
        if data_augmentation_cfg['rotation']:
            DA.rotate()
        
        self.geo_aug = DA.elastic_distortion(alpha=data_augmentation_cfg['elastic-transform']['alpha'],
                                            sigma=data_augmentation_cfg['elastic-transform']['sigma'],
                                            alpha_affine=data_augmentation_cfg['elastic-transform']['alpha-affine']
                       ).compile(images='x', labels='y,mask', fields='alpha', to_torch=True)
        self.factor = factor
        self._data_length = len(self.data)

    def __len__(self):
        return self._data_length * self.factor

    def __getitem__(self, i):
        i = i % self._data_length
        img = self.data[i].transpose(1, 2, 0)
        cos_sin_alpha = self.cos_sin_alpha[i].transpose(1, 2, 0)
        return self.geo_aug(x=img, y=self.av[i], mask=self.mask[i], alpha=cos_sin_alpha)


class TestDataset(Dataset):
    def __init__(self, dataset, file):
        super(TestDataset, self).__init__()
        DATA = h5py.File(file, 'r')

        self.data = DATA.get(f'{dataset}/data')
        self.av = DATA.get(f'{dataset}/av')
        self.cos_sin_alpha = DATA.get(f'{dataset}/principal-angle')
        self.mask = DATA.get(f'{dataset}/mask')
        self.geo_aug = DataAugment().compile(images='x', labels='y,mask', fields='alpha', to_torch=True)
        self._data_length = len(self.data)

    def __len__(self):
        return self._data_length

    def __getitem__(self, i):
        img = self.data[i].transpose(1, 2, 0)
        cos_sin_alpha = self.cos_sin_alpha[i].transpose(1, 2, 0)
        return self.geo_aug(x=img, y=self.av[i], mask=self.mask[i], alpha=cos_sin_alpha)
