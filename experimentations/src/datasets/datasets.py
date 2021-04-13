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
    steered = cfg.model.steered
    train_dataset = cfg.training['training-dataset']
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

        DATA = h5py.File(file, 'r')
        self.x = DATA.get(f'{dataset}/data')
        self.y = DATA.get(f'{dataset}/av')
        self.mask = DATA.get(f'{dataset}/mask')
        data_fields = dict(images='x', labels='y,mask')
        if steered:
            self.cos_sin_alpha = DATA.get(f'{dataset}/principal-angle')
            data_fields['fields'] = 'alpha'
        
        DA = DataAugment().flip()
        if data_augmentation_cfg['rotation']:
            DA.rotate()
        if data_augmentation_cfg['elastic']:
            DA.elastic_distortion(alpha=data_augmentation_cfg['elastic-transform']['alpha'],
                                  sigma=data_augmentation_cfg['elastic-transform']['sigma'],
                                  alpha_affine=data_augmentation_cfg['elastic-transform']['alpha-affine']
                                  )
        
        self.geo_aug = DA.compile(**data_fields, to_torch=True)
        self.factor = factor
        self.steered = steered
        self._data_length = len(self.x)

    def __len__(self):
        return self._data_length * self.factor

    def __getitem__(self, i):
        i = i % self._data_length
        x = self.x[i].transpose(1, 2, 0)
        if self.steered:
            cos_sin_alpha = self.cos_sin_alpha[i].transpose(1, 2, 0)
            return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i], alpha=cos_sin_alpha)
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
            self.cos_sin_alpha = DATA.get(f'{dataset}/principal-angle')
            data_fields['fields'] = 'alpha'
            
        self.geo_aug = DataAugment().compile(**data_fields, to_torch=True)
        
        self.steered = steered
        self._data_length = len(self.x)

    def __len__(self):
        return self._data_length

    def __getitem__(self, i):
        x = self.x[i].transpose(1, 2, 0)
        if self.steered:
            cos_sin_alpha = self.cos_sin_alpha[i].transpose(1, 2, 0)
            return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i], alpha=cos_sin_alpha)
        else:
            return self.geo_aug(x=x, y=self.y[i], mask=self.mask[i])
