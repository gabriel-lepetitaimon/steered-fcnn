import numpy as np
import h5py
import os.path as P
from torch.utils.data import Dataset

from ..config import default_config, AttributeDict
from .data_augment import DataAugment


def create_generic_hdf_datasets(dataset_cfg: AttributeDict, file_path=None):
    from datasets import DEFAULT_DATA_PATH
    if file_path is None:
        file_path = DEFAULT_DATA_PATH
    hdf_path = P.join(file_path, dataset_cfg.file)

    # --- Sanity Check of cfg ---
    data_names = set()
    datamap = dataset_cfg.data
    for specs in datamap.values():
        if '{{' not in specs:
            data_names.add(specs)
        else:
            data_names.add([_.split('}}')[0].strip() for _ in specs.split('{{')])

    train_dataset = dataset_cfg.get(['training-dataset'], None)
    train_dataset = '' if train_dataset is None or train_dataset[-1] == '/' else train_dataset+'/'

    with h5py.File(hdf_path, 'r') as hf:
        if 'train' not in hf:
            raise TypeError('Invalid GenericHDF archive: the hdf file should contain a group "/train/".')
        check_dataset_exist(hf, '/train/'+train_dataset, data_names, hdf_path)
        if 'val' not in hf:
            raise TypeError('Invalid GenericHDF archive: the hdf file should contain a group "/val/".')
        check_dataset_exist(hf, '/val/'+train_dataset, data_names, hdf_path)

        if 'test' not in hf:
            raise TypeError('Invalid GenericHDF archive: the hdf file should contain a group "/test/".')

        test_datasets = dataset_cfg.get('test-datasets', None)
        if test_datasets is None:
            if any(n in hf['test'] for n in data_names):
                test_datasets = [""]
            else:
                test_datasets = tuple(hf['test'].keys())
        test_datasets = [d if d == '' or d[-1] == '/' else d+'/' for d in test_datasets]

        for d in test_datasets:
            check_dataset_exist(hf, '/test/' if d == "" else f'/test/{d}/', data_names, hdf_path)


def create_generic_hdf_dataset(datasets_cfg: AttributeDict, prefix: str, hdf_path: str):

    names = 'any'
    dataaugment_enabled = prefix == 'train'
    if prefix not in datasets_cfg:


    with h5py.File(hdf_path, 'r') as hf:
        if prefix not in hf:
            raise TypeError(f'Invalid GenericHDF archive: the hdf file should contain a group "{prefix}".')


def check_dataset_exist(hf, prefix, data_names, hdf_path):
    for p in prefix.split('/'):
        if p:
            hf = hf[p]

    missing_data_names = {n for n in data_names if n not in hf}
    if missing_data_names:
        raise ValueError(f'Thw following dataset were not found in the hdf archive "{hdf_path}":\n'
                         f' {missing_data_names}')

    invalid_data_names = {n for n in data_names if not isinstance(hf[n], h5py.Dataset)}
    if invalid_data_names:
        raise ValueError(f'Thw following node are not dataset in the hdf archive "{hdf_path}":\n'
                         f' {invalid_data_names}')


class GenericHRF(Dataset):
    def __init__(self, dataset, file, mapping, cache=False, factor=1, data_augmentation_cfg=None):
        super(GenericHRF, self).__init__()

        with h5py.File(file, 'r') as DATA:
            self.x = DATA.get(f'{dataset}/data')[:]
            if use_preprocess is False:
                self.x = self.x[:, 3:]
            elif use_preprocess == 'only':
                self.x = self.x[:, :3]
            self.y = DATA.get(f'{dataset}/av')[:]
            self.mask = DATA.get(f'{dataset}/mask')[:]
            data_fields = dict(images='x', labels='y,mask')

            if steered:
                if steered is True:
                    steered = 'vec-norm'
                if steered == 'vec':
                    self.steer = DATA.get(f'{dataset}/principal-vec-norm')[:]
                    data_fields['vectors'] = 'alpha'
                elif steered == 'vec-norm':
                    self.steer = DATA.get(f'{dataset}/principal-vec')[:]
                    data_fields['vectors'] = 'alpha'
                elif steered == 'angle':
                    data_fields['angles'] = 'alpha'
                    self.steer = DATA.get(f'{dataset}/principal-angle')[:]
                elif steered == 'all':
                    self.angle = DATA.get(f'{dataset}/principal-angle')[:]
                    self.vec = DATA.get(f'{dataset}/principal-vec-norm')[:]  # Through sigmoid
                    self.vec_norm = DATA.get(f'{dataset}/principal-vec')[:]
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
    def __init__(self, dataset, file, steered=True, use_preprocess=True):
        super(TestDataset, self).__init__()
        with h5py.File(file, 'r') as DATA:

            self.x = DATA.get(f'{dataset}/data')[:]
            if use_preprocess is False:
                self.x = self.x[:, 3:]
            elif use_preprocess == 'only':
                self.x = self.x[:, :3]
            self.y = DATA.get(f'{dataset}/av')[:]
            self.mask = DATA.get(f'{dataset}/mask')[:]
            data_fields = dict(images='x', labels='y,mask')

            if steered:
                if steered is True:
                    steered = 'vec-norm'
                if steered == 'vec':
                    self.steer = DATA.get(f'{dataset}/principal-vec-norm')[:]
                    data_fields['vectors'] = 'alpha'
                elif steered == 'vec-norm':
                    self.steer = DATA.get(f'{dataset}/principal-vec')[:]
                    data_fields['vectors'] = 'alpha'
                elif steered == 'angle':
                    data_fields['angles'] = 'alpha'
                    self.steer = DATA.get(f'{dataset}/principal-angle')[:]
                elif steered == 'all':
                    self.angle = DATA.get(f'{dataset}/principal-angle')[:]
                    self.vec = DATA.get(f'{dataset}/principal-vec-norm')[:]  # Through sigmoid
                    self.vec_norm = DATA.get(f'{dataset}/principal-vec')[:]
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
