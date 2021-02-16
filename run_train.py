import numpy as np
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_objective

from src.model import HemelingNet, HemelingRotNet
from src.classifier_net import BinaryClassifierNet, ExportValidation


def run_train():
    args, exp_config = parse_arguments()
    hp_cfg = exp_config['hyper-parameters']
    cfg_experiment = exp_config['experiment']
    setup_log(exp_config)
    trainD, validD, testD = load_dataset(exp_config)

    trial_path = cfg_experiment['trial-path']
    val_n_epoch = cfg_experiment['val-every-n-epoch']
    max_epoch = cfg_experiment['max-epoch']

    # ---  MODEL  ---
    net = setup_model(cfg_experiment)

    # ---  TRAIN  ---
    trainer_kwargs = {}
    if cfg_experiment['half-precision']:
        trainer_kwargs['amp_level']='O2'
        trainer_kwargs['precision']=16

    callbacks = []
    if cfg_experiment['early-stopping']['monitor'].lower() != 'none':
        if cfg_experiment['early-stopping']['monitor'].lower() == 'auto':
            cfg_experiment['early-stopping']['monitor'] = cfg_experiment['optimize']
        callbacks += [EarlyStopping(verbose=False, strict=False, **cfg_experiment['early-stopping'])]
        
    bestCP_acc = ModelCheckpoint(trial_path+'/best-acc', monitor='valid-acc', mode='max')
    bestCP_roc = ModelCheckpoint(trial_path+'/best-roc', monitor='valid-roc', mode='max')
    bestCP_iou = ModelCheckpoint(trial_path+'/best-iou', monitor='valid-iou', mode='max')
    
    callbacks += [bestCP_acc, bestCP_roc, bestCP_iou]
    callbacks += [ExportValidation({(0,0): 'black', (1,1): 'white', (1,0): 'orange', (0,1): 'apple_green'}, path=f'{trial_path}/')]
    trainer = pl.Trainer(gpus=args.gpu, callbacks=callbacks, 
                         max_epochs=int(np.ceil(max_epoch/val_n_epoch)*val_n_epoch),
                         check_val_every_n_epoch=val_n_epoch,
                         progress_bar_refresh_rate=0,
                         **trainer_kwargs)
    net.log('valid-acc', 0)
    trainer.fit(net, trainD, validD)
    
    best_score = float(bestCP_acc.best_model_score.cpu().numpy())
    mlflow.log_metric('best_score', best_score)
    mlflow.log_metric('best-roc', float(bestCP_roc.best_model_score.cpu().numpy()))
    mlflow.log_metric('best-iou', float(bestCP_iou.best_model_score.cpu().numpy()))
    report_objective(-best_score)
    mlflow.end_run()


def parse_arguments():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    #parser.add_argument('--gpu', help='specify which gpu should be used', default=1)
    parser.add_argument('--config', required=True,
                        help='config file with hyper parameters - in yaml format')
    args = parser.parse_args()

    exp_config = parse_config(args.config)

    exp_name = os.getenv('ORION_EXPERIMENT_NAME', 'test')
    subexp_name = exp_config['experiment']['name']

    trials = sorted(_ for _ in os.listdir('experiments/' + exp_name + '/') if _.startswith(subexp_name))
    if len(trials):
        trialID = int(trials[-1][len(subexp_name) + 1:]) + 1
    else:
        trialID = 1
    trialName = subexp_name + '-%03d' % trialID
    trialPath = 'experiments/' + exp_name + '/' + trialName + '/'
    os.mkdir(trialPath)

    exp_config['experiment']['trial-id'] = trialID
    exp_config['experiment']['trial-name'] = trialName
    exp_config['experiment']['trial-path'] = trialPath
    exp_config['experiment']['name'] = exp_name
    exp_config['experiment']['version'] = os.getenv('ORION_EXPERIMENT_VERSION', 0)

    return args, exp_config


def parse_config(cfg_file):
    import yaml
    from src.utils import recursive_dict_update, recursive_dict_map

    with open('default_exp_config.yaml', 'r') as f:
        default_exp_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(cfg_file, 'r') as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)
    exp_config = recursive_dict_map(exp_config,
                                    lambda k, v: v if not isinstance(v, str) or not v.startswith('orion~') else None)
    recursive_dict_update(default_exp_config, exp_config)

    return default_exp_config


def setup_log(exp_config):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(exp_config['experiment']['name'])
    mlflow.pytorch.autolog(log_models=False)

    mlflow.start_run(run_name=exp_config['experiment']['trial-name'])
    mlflow.log_param('sub-experiment', exp_config['experiment']['name'])
    for k, v in exp_config['model'].items():
        mlflow.log_param(f'Model.{k}', v)
    for k, v in exp_config['data-augmentation'].items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                mlflow.log_param(f'DA.{k} {k1}', v1)
        else:
            mlflow.log_param(f'DA.{k}', v)

def load_dataset(exp_config):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import Dataset, DataLoader
    import cv2

    da_config = exp_config['data-augmentation']

    class TrainDataset(Dataset):
        def __init__(self, dataset, file, factor=1):
            super(TrainDataset, self).__init__()
            import h5py
            DATA = h5py.File(file, 'r')

            self.data = DATA.get(f'{dataset}/data')
            self.av = DATA.get(f'{dataset}/av')
            self.field = DATA.get(f'{dataset}/radial-field')
            self.mask = DATA.get(f'{dataset}/mask')

            self.geo_aug = A.Compose([
                A.PadIfNeeded(1024, 1024, value=0, border_mode=cv2.BORDER_CONSTANT),
                A.RandomCrop(da_config['crop-size'], da_config['crop-size']),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=(-180, 180)),
                A.ElasticTransform(alpha=da_config['elastic-transform']['alpha'],
                                   sigma=da_config['elastic-transform']['sigma'],
                                   alpha_affine=da_config['elastic-transform']['alpha-affine'],
                                   border_mode=cv2.BORDER_CONSTANT, p=.9),
                A.VerticalFlip(p=0.5),
                ToTensorV2()
            ])
            self.factor = factor
            self._data_length = len(self.data)

        def __len__(self):
            return self._data_length * self.factor

        def __getitem__(self, i):
            i = i % self._data_length
            img = np.concatenate(
                    [self.data[i].transpose(1, 2, 0),
                     self.field[i].transpose(1, 2, 0)],
                    axis=2)
            m = self.av[i]+self.mask[i]*16
            d = self.geo_aug(image=img, mask=m)
            r = {'x': d['image'][:6],
                 'principal_direction': d['image'][6:],
                 'y': d['mask']%16,
                 'mask': d['mask']//16}
            return r

    class TestDataset(Dataset):
        def __init__(self, dataset, file='DATA/vessels.h5'):
            super(TestDataset, self).__init__()
            import h5py
            DATA = h5py.File(file, 'r')
            
            self.data = DATA.get(f'{dataset}/data')
            self.av = DATA.get(f'{dataset}/av')
            self.field = DATA.get(f'{dataset}/radial-field')
            self.mask = DATA.get(f'{dataset}/mask')
            self.geo_aug = A.Compose([A.PadIfNeeded(1000, 1000, value=0, border_mode=cv2.BORDER_CONSTANT),
                                      ToTensorV2()])
            self._data_length = len(self.data)

        def __len__(self):
            return self._data_length

        def __getitem__(self, i):
            img = np.concatenate(
                [self.data[i].transpose(1,2,0),
                 self.field[i].transpose(1,2,0)],
                axis=2)
            m = self.av[i]+self.mask[i]*16
            d = self.geo_aug(image=img, mask=m)
            r = {'x': d['image'][:6],
                 'principal_direction': d['image'][6:],
                 'y': d['mask']%16,
                 'mask': d['mask']//16}
            return r

    batch_size=exp_config['hyper-parameters']['batch-size']
    train_dataset = exp_config['experiment']['training-dataset']
    dataset_file = exp_config['experiment']['dataset-file']
    trainD = DataLoader(TrainDataset('train/'+train_dataset, file=dataset_file, factor=8*3),
                        pin_memory=True, shuffle=True,
                        batch_size=batch_size,
                        num_workers=batch_size)
    validD = DataLoader(TestDataset('val/'+train_dataset, file=dataset_file),
                        pin_memory=True, num_workers=6, batch_size=6)
    testD = {_: DataLoader(TestDataset('test/'+_, file=dataset_file),
                           pin_memory=True, num_workers=6, batch_size=6)
             for _ in ('MESSIDOR', 'HRF', 'DRIVE')}
    return trainD, validD, testD


def setup_model(cfg):
    model_cfg = cfg['model']
    if model_cfg['rot-eq']:
        model = HemelingRotNet(6, principal_direction=1, nfeatures_base=model_cfg['nfeatures-base'],
                               half_kernel_height=model_cfg['half-kernel-height'],
                               padding=model_cfg['padding'],
                               depth=model_cfg['depth'],
                               p_dropout=model_cfg['drop-out'],
                               rotconv_squeeze=model_cfg['rotconv-squeeze'],
                               static_principal_direction=model_cfg['static-principal-direction'],
                               principal_direction_smooth=model_cfg['principal-direction-smooth'],
                               principal_direction_hessian_threshold=model_cfg['principal-direction-hessian-threshold'])
    else:
        model = HemelingNet(6, p_dropout=model_cfg['drop-out'], 
                               nfeatures_base=model_cfg['nfeatures-base'],
                               padding=model_cfg['padding'],
                               half_kernel_height=model_cfg['half-kernel-height'])
    net = BinaryClassifierNet(model=model, loss=cfg['hyper-parameters']['loss'], optimizer=cfg['hyper-parameters']['optimizer'], lr=cfg['hyper-parameters']['lr'])
    return net


if __name__ == '__main__':
    run_train()
