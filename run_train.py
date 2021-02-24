import numpy as np
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_objective
from pytorch_lightning.utilities.cloud_io import load as pl_load

from src.model import HemelingNet, HemelingRotNet
from src.classifier_net import BinaryClassifierNet, ExportValidation
from src.utils import AttributeDict


def run_train():
    cfg = parse_arguments()
    args = cfg['script-arguments']
    tmp = setup_log(cfg)
    tmp_path = tmp.name

    trainD, validD, testD = load_dataset(cfg)

    val_n_epoch = cfg.training['val-every-n-epoch']
    max_epoch = cfg.training['max-epoch']

    # ---  MODEL  ---
    model = setup_model(cfg['model'])

    # ---  TRAIN  ---
    trainer_kwargs = {}
    hyper_params = cfg['hyper-parameters']
    net = BinaryClassifierNet(model=model, loss=hyper_params['loss'],
                              optimizer=hyper_params['optimizer'],
                              lr=hyper_params['lr'],
                              p_dropout=hyper_params['drop-out'])
    if cfg.training['half-precision']:
        trainer_kwargs['amp_level'] = 'O2'
        trainer_kwargs['precision'] = 16

    callbacks = []
    if cfg.training['early-stopping']['monitor'].lower() != 'none':
        if cfg.training['early-stopping']['monitor'].lower() == 'auto':
            cfg.training['early-stopping']['monitor'] = cfg.training['optimize']
        callbacks += [EarlyStopping(verbose=False, strict=False, **cfg.training['early-stopping'])]

    modelCheckpoints = {}
    for metric in ('val-acc', 'val-roc', 'val-iou'):
        checkpoint = ModelCheckpoint(tmp_path, filename='best-'+metric, monitor=metric, mode='max')
        modelCheckpoints[metric] = checkpoint
        callbacks.append(checkpoint)

    trainer = pl.Trainer(gpus=args.gpus, callbacks=callbacks,
                         max_epochs=int(np.ceil(max_epoch/val_n_epoch)*val_n_epoch),
                         check_val_every_n_epoch=val_n_epoch,
                         progress_bar_refresh_rate=1 if args.debug else 0,
                         **trainer_kwargs)
    net.log(cfg.training['optimize'], 0)
    trainer.fit(net, trainD, validD)

    for metric_name, checkpoint in modelCheckpoints.items():
        metric_value = float(checkpoint.best_model_score.cpu().numpy())
        mlflow.log_metric('best-'+metric_name, metric_value)
        if metric_name == cfg.training['optimize']:
            report_objective(-metric_value)
            state_dict = pl_load(checkpoint.best_model_path)['state_dict']
            net.load_state_dict(state_dict)

    net.eval()
    tester = pl.Trainer(gpus=args.gpus,
                        callbacks=[ExportValidation({(0,0): 'black', (1,1): 'white', (1,0): 'orange', (0,1): 'apple_green'}, path=tmp_path+'/samples')],
                        )
    net.testset_names, testD = list(zip(*testD.items()))

    tester.test(net, testD)
    mlflow.log_artifacts(tmp.name)
    mlflow.end_run()
    tmp.cleanup()


def parse_arguments():
    import argparse
    import os

    # --- PARSE ARGS & ENVIRONNEMENTS VARIABLES ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='config file with hyper parameters - in yaml format')
    parser.add_argument('--debug', help='Debug trial (not logged into orion)',
                        default=bool(os.getenv('TRIAL_DEBUG', False)))
    parser.add_argument('--gpus', help='list of gpus to use for this trial',
                        default=os.getenv('TRIAL_GPUS',None))
    parser.add_argument('--tmp-dir', help='Directory where the trial temporary folders will be stored.',
                        default=os.getenv('TRIAL_TMP_DIR',None))
    args = parser.parse_args()

    # --- PARSE CONFIG ---
    cfg = parse_config(args.config)

    # Save scripts arguments
    script_args = cfg['script-arguments']
    for k, v in vars(args).items():
        if v is not None:
            script_args[k] = v
    # Save trial info
    cfg['trial'] = AttributeDict(id=int(os.getenv('TRIAL_ID'), 0),
                                 name=os.getenv('ORION_EXPERIMENT_NAME', 'trial-name'),
                                 version=os.getenv('ORION_EXPERIMENT_VERSION', 0))
    return cfg


def parse_config(cfg_file):
    with open('orion_config.yaml', 'r') as f:
        orion_config = AttributeDict.from_yaml(f)
    with open('global_config.yaml', 'r') as f:
        global_config = AttributeDict.from_yaml(f)
        global_config['orion'] = orion_config
    with open('default_exp_config.yaml', 'r') as f:
        global_config.recursive_update(AttributeDict.from_yaml(f))
    with open(cfg_file, 'r') as f:
        exp_config = AttributeDict.from_yaml(f)
    exp_config = exp_config.filter(lambda k, v: not (isinstance(v, str) and v.startswith('orion~')))
    return global_config.recursive_update(exp_config)


def setup_log(cfg):
    import tempfile
    import shutil
    from os.path import join
    from mlflow.tracking import MlflowClient

    # --- SETUP MLFOW ---
    mlflow.set_tracking_uri(cfg['mlflow']['uri'])
    mlflow.set_experiment(cfg['experiment']['name'] if not cfg['script-arguments'].debug else 'DEBUG_RUNS')
    mlflow.pytorch.autolog(log_models=False)
    tags = cfg.experiment.tags.to_dict()
    tags['subexp'] = cfg.experiment['sub-experiment']
    tags['subexpID'] = str(cfg.experiment['sub-experiment-id'])
    mlflow.start_run(run_name=cfg.trial.name, tags=tags)

    # --- CREATE TMP ---
    tmp = tempfile.TemporaryDirectory(dir=cfg['script-arguments']['tmp-dir'])

    # --- SAVE CFG ---
    shutil.copy(cfg['script-arguments'].config, join(tmp.name, 'cfg.yaml'))
    mlflow.log_artifact(join(tmp.name, 'cfg.yaml'))
    # Sanity check of artifact saving
    client = MlflowClient()
    artifacts = client.list_artifacts(mlflow.active_run().info.run_id)
    if len(artifacts) != 1 or artifacts[0].path != 'cfg.yaml':
        raise RuntimeError('The sanity check for storing artifacts failed.'
                           'Interrupting the script before the training starts.')

    with open(join(tmp.name, 'cfg_extended.yaml'), 'w') as f:
        cfg.to_yaml(f)

    mlflow.log_param('sub-experiment', cfg.experiment['sub-experiment'])
    if cfg.experiment['sub-experiment-id']:
        mlflow.log_param('sub-experiment-id', cfg.experiment['sub-experiment-id'])
    for k, v in cfg.trial.items():
        mlflow.log_param('trial.'+k, v)

    for k, v in cfg['model'].items():
        mlflow.log_param(f'model.{k}', v)
    for k, v in cfg['data-augmentation'].items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                mlflow.log_param(f'DA.{k} {k1}', v1)
        else:
            mlflow.log_param(f'DA.{k}', v)
    mlflow.log_param('dropout', cfg['hyper-parameters']['drop-out'])
    mlflow.log_param('training.file', cfg.training['dataset-file'])
    mlflow.log_param('training.dataset', cfg.training['training-dataset'])

    return tmp


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
                A.PadIfNeeded(800, 800, value=0, border_mode=cv2.BORDER_CONSTANT),
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
            img = [self.data[i].transpose(1, 2, 0)]
            princ_dir_mode = cfg['model']['static-principal-direction']
            if princ_dir_mode:
                princ_dir = self.field[i]
                if princ_dir_mode == 'normalized':
                    princ_dir /= np.sqrt(princ_dir[0]**2+princ_dir[1]**2) + 1e-5
                img += [princ_dir.transpose(1, 2, 0)]
            img = np.concatenate(img, axis=2)
            m = 1*(self.av[i]!=0)+self.mask[i]*16
            d = self.geo_aug(image=img, mask=m)
            r = {'x': d['image'][:6],
                 'y': (d['mask']%16).int(),
                 'mask': d['mask']//16}
            if princ_dir_mode:
                r['principal_direction'] = d['image'][6:]
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
            self.geo_aug = A.Compose([A.PadIfNeeded(800, 800, value=0, border_mode=cv2.BORDER_CONSTANT),
                                      ToTensorV2()])
            self._data_length = len(self.data)

        def __len__(self):
            return self._data_length

        def __getitem__(self, i):
            img = np.concatenate(
                [self.data[i].transpose(1,2,0),
                 self.field[i].transpose(1,2,0)],
                axis=2)
            m = 1*(self.av[i]!=0)+self.mask[i]*16
            d = self.geo_aug(image=img, mask=m)
            r = {'x': d['image'][:6],
                 'principal_direction': d['image'][6:],
                 'y': (d['mask'] % 16).int(),
                 'mask': d['mask']//16}
            return r

    batch_size=cfg['hyper-parameters']['batch-size']
    train_dataset = cfg.training['training-dataset']
    dataset_file = cfg.training['dataset-file']
    trainD = DataLoader(TrainDataset('train/'+train_dataset, file=dataset_file, factor=cfg.training['training-dataset-factor']),
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
                               rotconv_squeeze=model_cfg['rotconv-squeeze'],
                               static_principal_direction=model_cfg['static-principal-direction'],
                               principal_direction_smooth=model_cfg['principal-direction-smooth'],
                               principal_direction_hessian_threshold=model_cfg['principal-direction-hessian-threshold'],
                               sym_kernel=model_cfg['sym-kernel'])
    else:
        model = HemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                               padding=model_cfg['padding'],
                               half_kernel_height=model_cfg['half-kernel-height'])
    return model


if __name__ == '__main__':
    run_train()
