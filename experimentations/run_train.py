import numpy as np
import mlflow
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_objective
from pytorch_lightning.utilities.cloud_io import load as pl_load
import os
import os.path as P
from json import dump

from src.datasets import load_dataset
from src.trainer import BinaryClassifierNet, ExportValidation
from src.trainer.loggers import setup_log
from steered_cnn.models import setup_model


def run_train(**opt):
    cfg = parse_arguments(opt)
    args = cfg['script-arguments']
    tmp = setup_log(cfg)
    tmp_path = tmp.name
    
    if isinstance(cfg.training['seed'], int):
        torch.manual_seed(cfg.training['seed'])
        np.random.seed(cfg.training['seed'])

    trainD, validD, testD = load_dataset(cfg)

    val_n_epoch = cfg.training['val-every-n-epoch']
    max_epoch = cfg.training['max-epoch']

    # ---  MODEL  ---
    model = setup_model(cfg['model'])

    # ---  TRAIN  ---
    r_code = 10
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

    checkpointed_metrics = ['val-acc', 'val-roc', 'val-iou']
    modelCheckpoints = {}
    for metric in checkpointed_metrics:
        checkpoint = ModelCheckpoint(dirpath=tmp_path + '/', filename='best-' + metric, monitor=metric, mode='max')
        modelCheckpoints[metric] = checkpoint
        callbacks.append(checkpoint)
        
    trainer = pl.Trainer(gpus=args.gpus, callbacks=callbacks,
                         max_epochs=int(np.ceil(max_epoch / val_n_epoch) * val_n_epoch),
                         check_val_every_n_epoch=val_n_epoch,
                         accumulate_grad_batches=cfg['hyper-parameters']['accumulate-gradient-batch'],
                         progress_bar_refresh_rate=1 if args.debug else 0,
                         **trainer_kwargs)
    net.log(cfg.training['optimize'], 0)
    try:
        trainer.fit(net, trainD, validD)
    except KeyboardInterrupt:
        r_code = 1

    
    # --- TEST ---
    reported_metric = cfg.training['optimize']
    if reported_metric not in modelCheckpoints:
        print(f'Invalid optimized metric {reported_metric}, optimizing {checkpointed_metrics[0]} instead.')
        reported_metric = reported_metric[0]
    for metric_name, checkpoint in modelCheckpoints.items():
        metric_value = float(checkpoint.best_model_score.cpu().numpy())
        mlflow.log_metric('best-' + metric_name, metric_value)
        if metric_name == reported_metric:
            reported_value = -metric_value
            state_dict = pl_load(checkpoint.best_model_path)['state_dict']
            net.load_state_dict(state_dict)

    net.eval()
    
    if 'av' in cfg.training['dataset-file']:
        cmap = {(0, 0): 'blue', (1, 1): 'red', (1, 0): 'cyan', (0, 1): 'pink', 'default': 'lightgray'}
    else:
        cmap = {(0, 0): 'black', (1, 1): 'white', (1, 0): 'orange', (0, 1): 'greenyellow', 'default': 'lightgray'}
        
    
    tester = pl.Trainer(gpus=args.gpus,
                        callbacks=[ExportValidation(cmap, path=tmp_path + '/samples')],)
    net.testset_names, testD = list(zip(*testD.items()))
    tester.test(net, testD)
    
    # --- LOG ---
    report_objective(reported_value)
    mlflow.log_artifacts(tmp.name)
    mlflow.end_run()
    tmp.cleanup()

    with open(P.join(cfg['script-arguments']['tmp-dir'], f'{cfg.trial.name}-{cfg.trial.id}.json'), 'w') as f:
        json = {'rcode': r_code}
        dump(json, f)


def parse_arguments(opt=None):
    import argparse
    from src.config import parse_config, AttributeDict

    # --- PARSE ARGS & ENVIRONNEMENTS VARIABLES ---
    if not opt:
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


# def setup_model(model_cfg, old=False):
#     from steered_cnn.models import HemelingNet, SteeredHemelingNet, OldHemelingNet
#     if model_cfg['steered']:
#         model = SteeredHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
#                                    padding=model_cfg['padding'],
#                                    depth=model_cfg['depth'],
#                                    batchnorm=model_cfg['batchnorm'],
#                                    upsample=model_cfg['upsample'],
#                                    attention=model_cfg['steered'] == 'attention',
#                                    static_principal_direction=model_cfg['static-principal-direction'])
#     else:
#         if old:
#             model = OldHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
#                                 padding=model_cfg['padding'],
#                                 half_kernel_height=model_cfg['half-kernel-height'])
#         else:
#             model = HemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
#                                 padding=model_cfg['padding'],
#                                 depth=model_cfg['depth'],
#                                 batchnorm=model_cfg['batchnorm'],
#                                 half_kernel_height=model_cfg['half-kernel-height'])
#     return model


if __name__ == '__main__':
    run_train()
