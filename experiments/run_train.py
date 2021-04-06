import sys
import numpy as np
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_objective
from pytorch_lightning.utilities.cloud_io import load as pl_load

import sys
import os.path as P
sys.path.insert(0, P.abspath(P.join(P.dirname(__file__), '../')))


def run_train(opt=None):
    from .datasets import load_dataset
    from .trainer.classifier_net import BinaryClassifierNet, ExportValidation
    cfg = parse_arguments(opt)
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
        checkpoint = ModelCheckpoint(dirpath=tmp_path + '/', filename='best-' + metric, monitor=metric, mode='max')
        modelCheckpoints[metric] = checkpoint
        callbacks.append(checkpoint)

    trainer = pl.Trainer(gpus=args.gpus, callbacks=callbacks,
                         max_epochs=int(np.ceil(max_epoch / val_n_epoch) * val_n_epoch),
                         check_val_every_n_epoch=val_n_epoch,
                         progress_bar_refresh_rate=1 if args.debug else 0,
                         **trainer_kwargs)
    net.log(cfg.training['optimize'], 0)
    trainer.fit(net, trainD, validD)

    for metric_name, checkpoint in modelCheckpoints.items():
        metric_value = float(checkpoint.best_model_score.cpu().numpy())
        mlflow.log_metric('best-' + metric_name, metric_value)
        if metric_name == cfg.training['optimize']:
            report_objective(-metric_value)
            state_dict = pl_load(checkpoint.best_model_path)['state_dict']
            net.load_state_dict(state_dict)

    net.eval()
    tester = pl.Trainer(gpus=args.gpus,
                        callbacks=[ExportValidation(
                            {(0, 0): 'black', (1, 1): 'white', (1, 0): 'orange', (0, 1): 'apple_green'},
                            path=tmp_path + '/samples')],
                        )
    net.testset_names, testD = list(zip(*testD.items()))

    tester.test(net, testD)
    mlflow.log_artifacts(tmp.name)
    mlflow.end_run()
    tmp.cleanup()
    sys.exit(2)


def parse_arguments(opt=None):
    import argparse
    import os
    from .config import parse_config
    from .utils.collections import AttributeDict

    # --- PARSE ARGS & ENVIRONNEMENTS VARIABLES ---
    if opt is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True,
                            help='config file with hyper parameters - in yaml format')
        parser.add_argument('--debug', help='Debug trial (not logged into orion)',
                            default=bool(os.getenv('TRIAL_DEBUG', False)))
        parser.add_argument('--gpus', help='list of gpus to use for this trial',
                            default=os.getenv('TRIAL_GPUS', None))
        parser.add_argument('--tmp-dir', help='Directory where the trial temporary folders will be stored.',
                            default=os.getenv('TRIAL_TMP_DIR', None))
        args = vars(parser.parse_args())
    else:
        args = {'config': opt.get('config'),
                'debug': opt.get('debug', False),
                'gpus': opt.get('gpus', None),
                'tmp-dir': opt.get('tmp-dir', None)}
        args = AttributeDict.from_dict(args)

    # --- PARSE CONFIG ---
    cfg = parse_config(args['config'])

    # Save scripts arguments
    script_args = cfg['script-arguments']
    for k, v in args.items():
        if v is not None:
            script_args[k] = v
    # Save trial info
    cfg['trial'] = AttributeDict(id=int(os.getenv('TRIAL_ID', 0)),
                                 name=os.getenv('ORION_EXPERIMENT_NAME', 'trial-name'),
                                 version=os.getenv('ORION_EXPERIMENT_VERSION', 0))
    if script_args.debug:
        cfg.training['max-epoch'] = 1
    return cfg


def setup_log(cfg):
    import tempfile
    import shutil
    from os.path import join
    import os
    from mlflow.tracking import MlflowClient

    # --- SETUP MLFOW ---
    mlflow.set_tracking_uri(cfg['mlflow']['uri'])
    mlflow.set_experiment(cfg['experiment']['name'] if not cfg['script-arguments'].debug else 'DEBUG_RUNS')
    mlflow.pytorch.autolog(log_models=False)
    tags = cfg.experiment.tags.to_dict()
    tags['subexp'] = cfg.experiment['sub-experiment']
    tags['subexpID'] = str(cfg.experiment['sub-experiment-id'])
    run_name = f"{cfg.experiment['sub-experiment']}{cfg.experiment['sub-experiment-id']}-{cfg.trial.id:02}"
    mlflow.start_run(run_name=run_name, tags=tags)

    # --- CREATE TMP ---
    os.makedirs(os.path.dirname(cfg['script-arguments']['tmp-dir']), exist_ok=True)
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
        mlflow.log_param('trial.' + k, v)

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


def setup_model(model_cfg, old=False):
    from lib.models import HemelingNet, SteeredHemelingNet, OldHemelingNet
    if model_cfg['steered']:
        model = SteeredHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                   padding=model_cfg['padding'],
                                   depth=model_cfg['depth'],
                                   static_principal_direction=model_cfg['static-principal-direction'])
    else:
        if old:
            model = OldHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                padding=model_cfg['padding'],
                                half_kernel_height=model_cfg['half-kernel-height'])
        else:
            model = HemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                padding=model_cfg['padding'],
                                half_kernel_height=model_cfg['half-kernel-height'])
    return model


if __name__ == '__main__':
    run_train()
