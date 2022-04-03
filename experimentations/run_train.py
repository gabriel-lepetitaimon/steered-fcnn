import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, DeviceStatsMonitor, StochasticWeightAveraging
from orion.client import report_objective
import os
import os.path as P
from json import dump

from src.config import parse_arguments
from src.datasets import load_dataset
from src.trainer import Binary2DSegmentation, ExportSegmentation, ExportClassification
from src.trainer.loggers import Logs
from steered_cnn.models import setup_model


def run_train(**opt):
    # --- Parse cfg ---
    cfg = parse_arguments(opt)
    args = cfg['script-arguments']

    # --- Set Seed --
    seed = cfg.training.get('seed', None)
    if seed == "random":
        seed = int.from_bytes(os.getrandom(32), 'little', signed=False)
    elif isinstance(seed, (tuple, list)):
        seed = seed[cfg.trial.ID % len(seed)]
    if isinstance(seed, int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg.training['seed'] = seed
    elif seed is not None:
        print(f"Seed can't be interpreted as int and will be ignored.")

    # --- Setup logs ---
    logs = Logs()
    logs.setup_log(cfg)
    tmp_path = logs.tmp_path

    try:
        # --- Setup dataset ---
        trainD, validD, testD = load_dataset(cfg)

        if args.debug:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            names = ['train', 'valid'] + ['test-'+k for k in testD.keys()]
            fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, row_titles=names)
            for i, dataloader in enumerate([trainD, validD]+[d for d in testD.values()]):
                sample = next(iter(dataloader))['x'].cpu().transpose(0, 1).flatten(1)
                for s, n in zip(sample, 'rgb'):
                    bins = 255
                    hist = torch.histc(s, bins=bins,)
                    vmin, vmax = sample.min().item(), sample.max().item()
                    w = (vmax-vmin)/(bins)/4
                    x = np.linspace(sample.min().item(), sample.max().item(), bins) + {'r': -w, 'g': 0, 'b': w}[n]
                    fig.add_bar(x=x, y=hist, name=n, width=w,
                                marker_color={'r': '#ff0000', 'g': '#00ff00', 'b': '#0000ff'}[n], row=i+1, col=1)
            fig.update_layout(barmode='group', margin=dict(l=20, r=20, t=20, b=20), height=250*len(names))
            logs.log_plotly(f'hist/x', fig)

        ###################
        # ---  MODEL  --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
        sample = validD.dataset[0]
        model = setup_model(cfg['model'], n_in=sample['x'].shape[0],
                            n_out=1 if not torch.is_tensor(sample['y']) or sample['y'].ndim<=2 else sample['y'].shape[0], 
                            mode=cfg.training.get('mode', 'segment'))

        sample = None
        hyper_params = cfg['hyper-parameters']
        net = Binary2DSegmentation(model=model, loss=hyper_params['loss'],
                                   pos_weighted_loss=hyper_params['pos-weighted-loss'],
                                   soft_label=hyper_params['smooth-label'],
                                   earlystop_cfg=cfg['training']['early-stopping'],
                                   optimizer=hyper_params['optimizer'],
                                   lr=hyper_params['lr'] / hyper_params['accumulate-gradient-batch'],
                                   p_dropout=hyper_params['drop-out'],
                                   testset_names= list(testD.keys()))
        logs.log_miscs({'model': {
            'params': sum(p.numel() for p in net.parameters())
        }})

        ###################
        # ---  TRAIN  --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
        val_n_epoch = cfg.training['val-every-n-epoch']
        max_epoch = cfg.training['max-epoch']

        # Define r_code, a return code sended back to train_single.py.
        # (A run is considered successful if it returns 10 <= r <= 20. Otherwise orion is interrupted.)
        r_code = 10

        trainer_kwargs = {}
        if cfg.training['half-precision']:
            trainer_kwargs['precision'] = 16

        callbacks = []
        if cfg.training['early-stopping']['monitor'].lower() != 'none':
            if cfg.training['early-stopping']['monitor'].lower() == 'auto':
                cfg.training['early-stopping']['monitor'] = cfg.training['optimize']
            earlystop = EarlyStopping(verbose=False, strict=False, **cfg.training['early-stopping'])
            callbacks += [earlystop]
        else:
            earlystop = None
        callbacks += [LearningRateMonitor(logging_interval='epoch')]
        if args.debug:
            callbacks += [DeviceStatsMonitor()]

        if cfg.training.get('SWA', False):
            callbacks += [StochasticWeightAveraging()]

        checkpointed_metrics = ['val-kappa']
        modelCheckpoints = {}
        for metric in checkpointed_metrics:
            checkpoint = ModelCheckpoint(dirpath=tmp_path + '/', filename='best-'+metric+'-{epoch}', monitor=metric, mode='max')
            modelCheckpoints[metric] = checkpoint
            callbacks.append(checkpoint)

        trainer = pl.Trainer(gpus=args.gpus, callbacks=callbacks, logger=logs.loggers,
                             max_epochs=int(np.ceil(max_epoch / val_n_epoch) * val_n_epoch),
                             check_val_every_n_epoch=val_n_epoch,
                             accumulate_grad_batches=cfg['hyper-parameters']['accumulate-gradient-batch'],
                             progress_bar_refresh_rate=1 if args.debug else 0,
                             **trainer_kwargs)

        try:
            trainer.fit(net, trainD, validD)
        except KeyboardInterrupt:
            r_code = 1  # Interrupt Orion

        logs.log_metric('last-epoch', earlystop.stopped_epoch if earlystop is not None else max_epoch)
        logs.log_misc('CPU avg idle time train', trainD.avg_idle())
        logs.log_misc('avg time per iter train', trainD.avg_total())
        logs.log_misc('CPU avg idle time valid', validD.avg_idle())
        logs.log_misc('avg time per iter valid', validD.avg_total())

        ################
        # --- TEST --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾ #
        reported_metric = cfg.training['optimize']
        best_ckpt = None
        if reported_metric not in modelCheckpoints:
            print('\n!!!!!!!!!!!!!!!!!!!!')
            print(f'>> Invalid optimized metric {reported_metric}, optimizing {checkpointed_metrics[0]} instead.')
            print('')
            reported_metric = checkpointed_metrics[0]
        for metric_name, checkpoint in modelCheckpoints.items():
            metric_value = float(checkpoint.best_model_score.cpu().numpy())
            logs.log_metrics({'best-' + metric_name: metric_value,
                              f'best-{metric_name}-epoch': float(checkpoint.best_model_path[:-5].rsplit('-', 1)[1][6:])})
            if metric_name == reported_metric:
                best_ckpt = checkpoint
                reported_value = -metric_value

        callbacks = []
        if cfg.training.mode == 'segment':
            if 'av' in cfg.training['dataset-file']:
                cmap = {(0, 0): 'blue', (1, 1): 'red', (1, 0): 'cyan', (0, 1): 'pink', 'default': 'lightgray'}
            else:
                cmap = {(0, 0): 'black', (1, 1): 'white', (1, 0): 'orange', (0, 1): 'greenyellow', 'default': 'lightgray'}
            callbacks += [ExportSegmentation(cmap, path=tmp_path + '/samples', dataset_names=net.testset_names)]
        elif cfg.training.mode == 'classification':
            callbacks += [ExportClassification(n=5, path=tmp_path + '/test.png')]
            
        testD = list(testD.values())
        tester = pl.Trainer(gpus=args.gpus, logger=logs.loggers,
                            callbacks=callbacks,
                            progress_bar_refresh_rate=1 if args.debug else 0,)
        tester.test(net, testD, ckpt_path=best_ckpt.best_model_path)

        report_objective(reported_value)
    except:
        import traceback
        r_code = -1
        logs.log_misc('exception', traceback.format_exc().splitlines())
        traceback.print_exc()

    ###############
    # --- LOG --- #
    # ‾‾‾‾‾‾‾‾‾‾‾ #
    logs.save_cleanup()

    # Store data in a json file to send info back to train_single.py script.
    with open(P.join(cfg['script-arguments']['tmp-dir'], f'result.json'), 'w') as f:
        json = {'r_code': r_code}
        dump(json, f)
        print("WRITING JSON AT: ", P.join(cfg['script-arguments']['tmp-dir'], f'result.json'))


def leg_setup_model(model_cfg, old=False):
    from steered_cnn.models import HemelingNet, SteeredHemelingNet, OldHemelingNet
    if model_cfg['steered']:
        model = SteeredHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                   padding=model_cfg['padding'],
                                   depth=model_cfg['depth'],
                                   batchnorm=model_cfg['batchnorm'],
                                   upsample=model_cfg['upsample'],
                                   attention=model_cfg['steered'] == 'attention')
    else:
        if old:
            model = OldHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                padding=model_cfg['padding'],
                                half_kernel_height=model_cfg['half-kernel-height'])
        else:
            model = HemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                padding=model_cfg['padding'],
                                depth=model_cfg['depth'],
                                batchnorm=model_cfg['batchnorm'],
                                half_kernel_height=model_cfg['half-kernel-height'])
    return model


if __name__ == '__main__':
    run_train()
