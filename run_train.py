import numpy as np
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_objective

from src.model import HemelingNet, HemelingRotNet
from src.classifier_net import BinaryClassifierNet, ExportValidation
from src.utils import AttributeDict


def run_train():
    cfg = parse_arguments()
    cfg_experiment = cfg['experiment']
    tmp_path = setup_log(cfg)
    trainD, validD, testD = load_dataset(cfg)

    trial_path = cfg_experiment['trial-path']
    val_n_epoch = cfg_experiment['val-every-n-epoch']
    max_epoch = cfg_experiment['max-epoch']

    # ---  MODEL  ---
    model = setup_model(cfg['model'])

    # ---  TRAIN  ---
    trainer_kwargs = {}
    hyper_params = cfg['hyper-parameters']
    net = BinaryClassifierNet(model=model, loss=hyper_params['loss'],
                              optimizer=hyper_params['optimizer'],
                              lr=hyper_params['lr'],
                              p_dropout=hyper_params['dropout'])
    if cfg_experiment['half-precision']:
        trainer_kwargs['amp_level'] = 'O2'
        trainer_kwargs['precision'] = 16

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
    import tempfile

    # --- PARSE ARGS & ENVIRONNEMENTS VARIABLES ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='config file with hyper parameters - in yaml format')
    parser.add_argument('--debug', help='Debug trial (not logged into orion)',
                        default=os.getenv('TRIAL_DEBUG', False))
    parser.add_argument('--gpus', help='list of gpus to use for this trial',
                        default=os.getenv('TRIAL_GPUS'))
    args = parser.parse_args()

    # --- PARSE CONFIG ---
    cfg = parse_config(args.config)

    # Save scripts arguments
    script_args = cfg['script-arguments']
    script_args['gpus'] = args.gpus
    script_args['debug'] = args.debug

    # Save trial info
    cfg['trial'] = AttributeDict(id=os.getenv('TRIAL_ID', 0),
                                 name=os.getenv('ORION_EXPERIMENT_NAME', 'trial-name'),
                                 version=os.getenv('ORION_EXPERIMENT_VERSION', 0))
    return cfg


def parse_config(cfg_file):
    with open('global_config.yaml', 'r') as f:
        default_exp_config = AttributeDict.from_yaml(f)
    with open('default_exp_config.yaml', 'r') as f:
        default_exp_config.recursive_update(AttributeDict.from_yaml(f))
    with open(cfg_file, 'r') as f:
        exp_config = AttributeDict.from_yaml(f)
    exp_config = exp_config.filter(lambda k, v: not (isinstance(v, str) and v.startswith('orion~')))
    return default_exp_config.recursive_update(exp_config)


def setup_log(cfg):
    import tempfile

    mlflow.set_tracking_uri(cfg['mlflow']['uri'])
    mlflow.set_experiment(cfg['experiment']['name'])
    mlflow.pytorch.autolog(log_models=False)

    tempfile.TemporaryDirectory()

    mlflow.start_run(run_name=cfg.trial.name)
    mlflow.log_param('sub-experiment', cfg.experiment['sub-experiment'])
    if cfg.experiment['sub-experiment-id']:
        mlflow.log_param('sub-experiment-id', cfg.experiment['sub-experiment-id'])
    for k, v in cfg.trial:
        mlflow.log_param('trial.'+k, v)

    for k, v in cfg['model'].items():
        mlflow.log_param(f'model.{k}', v)
    for k, v in cfg['data-augmentation'].items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                mlflow.log_param(f'DA.{k} {k1}', v1)
        else:
            mlflow.log_param(f'DA.{k}', v)

def load_dataset(cfg):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import Dataset, DataLoader
    import cv2

    da_config = cfg['data-augmentation']

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

    batch_size=cfg['hyper-parameters']['batch-size']
    train_dataset = cfg['experiment']['training-dataset']
    dataset_file = cfg['experiment']['dataset-file']
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


def setup_model(model_cfg):
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
    return model


if __name__ == '__main__':
    run_train()
