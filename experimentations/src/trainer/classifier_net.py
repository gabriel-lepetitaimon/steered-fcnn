from functools import partial

# import mlflow
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as M
import segmentation_models_pytorch as smp

from steered_cnn.utils import clip_pad_center
from ..config import default_config


class Binary2DSegmentation(pl.LightningModule):
    def __init__(self, model, loss='binaryCE', pos_weighted_loss=False, optimizer=None, earlystop_cfg=None, lr=1e-3, p_dropout=0, soft_label=0):
        super().__init__()
        self.model = model
        self._metrics = torch.nn.ModuleDict()
        self.lr = lr
        self.p_dropout = p_dropout
        self.soft_label = soft_label
        
        if isinstance(loss, dict):
            loss_kwargs = loss
            loss = loss['type']
            del loss_kwargs['type']
        else:
            loss_kwargs = {}

        self.pos_weighted_loss = pos_weighted_loss
        if pos_weighted_loss:
            if loss == 'binaryCE':
                self._loss = lambda y_hat, y, weight: F.binary_cross_entropy_with_logits(y_hat, y.float(),
                                                                                         pos_weight=weight)
            else:
                raise ValueError(f'Invalid weighted loss function: "{loss}". (Only "binaryCE" is supported.)')
        else:
            if loss == 'dice':
                from .losses import binary_dice_loss
                self._loss = lambda y_hat, y: binary_dice_loss(torch.sigmoid(y_hat), y)
            elif loss == 'focalLoss':
                from .losses import focal_loss
                _loss = partial(focal_loss, gamma=loss_kwargs.get('gamma', 2))
                self._loss = lambda y_hat, y: _loss(torch.sigmoid(y_hat), y)
            elif loss == 'binaryCE':
                self._loss = lambda y_hat, y: F.binary_cross_entropy_with_logits(y_hat, y.float())
            elif loss == 'kappa':
                self._loss = smp.losses.JaccardLoss('binary', from_logits=False)
            else:
                raise ValueError(f'Unkown loss function: "{loss}". \n'
                                 f'Should be one of "dice", "focalLoss", "binaryCE".')

        if optimizer is None:
            optimizer = {'type': 'Adam'}
        self.optimizer = optimizer
        if earlystop_cfg is None:
            earlystop_cfg= default_config()['training']['early-stopping']
        self.earlystop_cfg = earlystop_cfg

        self.testset_names = None
        
    def loss_f(self, pred, target, weight=None):
        if self.soft_label:
            target = target.float()
            target *= 1-2*self.soft_label
            target += self.soft_label
        if self.pos_weighted_loss:
            return self._loss(pred, target, weight)
        else:
            return self._loss(pred, target)

    def compute_y_yhat(self, batch):
        x = batch['x']
        y = (batch['y'] != 0).int()
        y_hat = self.model(x, **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')}).squeeze(1)
        y = clip_pad_center(y, y_hat.shape)
        return y, y_hat

    def training_step(self, batch, batch_idx):
        y, y_hat = self.compute_y_yhat(batch)

        mask = None
        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_hat.shape)
            thr_mask = mask!=0
            y_hat = y_hat[thr_mask].flatten()
            y = y[thr_mask].flatten()
            if self.pos_weighted_loss:
                mask = mask[thr_mask].flatten()
        if self.pos_weighted_loss:
            loss = self.loss_f(y_hat, y, mask)
        else:
            loss = self.loss_f(y_hat, y)
        loss_value = loss.detach().cpu().item()
        self.log('train-loss', loss_value, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def _validate(self, batch, save_preds=False, save_loss=False):
        y, y_hat = self.compute_y_yhat(batch)
        if save_preds:
            y_pred = y_hat > 0
        
        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_hat.shape)
            y_hat = y_hat[mask != 0]
            y = y[mask != 0]

        y = y.flatten()
        y_hat = y_hat.flatten()

        r = {'target': y}
        if save_preds:
            r['preds'] = y_pred
        if save_loss:
            r['loss'] = self.loss_f(y_hat, y)
        r['probas'] = torch.sigmoid(y_hat)
        return r

    def update_metrics(self, prefix, probas, targets):
        try:
            metrics = self._metrics[prefix]
        except KeyError:
            thr = 0.5
            metrics = torch.nn.ModuleDict({
                'roc': M.AUROC(),
                'confmat': M.ConfusionMatrix(num_classes=2, threshold=thr),
                'acc': M.Accuracy(num_classes=2, threshold=thr),
                'kappa': M.CohenKappa(num_classes=2, threshold=thr),
            })
            self._metrics[prefix] = metrics

        if probas is not None:
            for k, m in metrics.items():
                m.update(probas, targets)

    def log_metrics(self, prefix, reset=True, discard_dataloaderidx=False):
        try:
            metrics = self._metrics[prefix]
        except KeyError:
            self.update_metrics(prefix, None, None)
            metrics = self._metrics[prefix]

        prefix = prefix+'-'

        for k, m in metrics.items():
            if discard_dataloaderidx:
                idx = self._current_dataloader_idx
                self._current_dataloader_idx = None
            v = m.compute()
            if k == 'confmat':
                confmat = v.cpu().item()
                self.log(prefix+'TN', confmat[0, 0])
                self.log(prefix+'TP', confmat[1, 1])
                self.log(prefix+'FN', confmat[1, 0])
                self.log(prefix+'FP', confmat[0, 1])
            else:
                self.log(prefix + k, v.cpu().item())

            if discard_dataloaderidx:
                self._current_dataloader_idx = idx

        if reset:
            for m in metrics.values():
                m.reset()

    def validation_step(self, batch, batch_idx):
        return self._validate(batch, save_preds=False, save_loss=False)

    def validation_step_end(self, outputs):
        self.update_metrics('val', outputs['probas'], outputs['target'])
        return outputs

    def validation_epoch_end(self, outputs) -> None:
        self.log_metrics('val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return {**self._validate(batch, save_preds=True, save_loss=False),
                **{'dataloader_idx': dataloader_idx}}

    def test_step_end(self, outputs):
        for probas, target, dataloader_idx in zip(outputs['probas'], outputs['target'], outputs['dataloader_idx']):
            prefix = 'test'
            if self.testset_names:
                prefix = self.testset_names[dataloader_idx]
            self.update_metrics(prefix, probas, target)
        return torch.cat(outputs['preds'])

    def test_epoch_end(self, outputs) -> None:
        prefixs = self.testset_names if self.testset_names else ['test']
        for prefix in prefixs:
            self.log_metrics(prefix, discard_dataloaderidx=True)

    def configure_optimizers(self):
        opt = self.optimizer
        if opt['type'].lower() in ('adam', 'adamax', 'adamw'):
            Adam = {'adam': torch.optim.Adam,
                    'adamax': torch.optim.Adamax,
                    'adamw': torch.optim.AdamW}[opt['type'].lower()]
            kwargs = {k: v for k, v in opt.items() if k in ('weight_decay', 'amsgrad', 'eps')}
            optimizer = Adam(self.parameters(), lr=self.lr, betas=(opt.get('beta', .9), opt.get('beta_sqr', .999)),
                             **kwargs)
        elif opt['type'].lower() == 'asgd':
            kwargs = {k: v for k, v in opt.items() if k in ('lambd', 'alpha', 't0', 'weight_decay')}
            optimizer = torch.optim.ASGD(self.parameters(), lr=self.lr, **kwargs)
        elif opt['type'].lower() == 'sgd':
            kwargs = {k: v for k, v in opt.items() if k in ('momentum', 'dampening', 'nesterov', 'weight_decay')}
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, **kwargs)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if opt['lr-decay-factor']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.earlystop_cfg['mode'],
                                                                   factor=opt['lr-decay-factor'],
                                                                   patience=self.earlystop_cfg['patience']/2,
                                                                   threshold=self.earlystop_cfg['min_delta'],
                                                                   min_lr=self.lr*opt['lr-decay-factor']**5)
            return {'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'frequency': 1, 'interval': 'epoch',
                        'monitor': self.earlystop_cfg['monitor'],

                    }}
        else:
            return optimizer

    def forward(self, *args, **kwargs):
        return torch.sigmoid(self.model(*args, **kwargs))

    def test(self, datasets):
        if isinstance(datasets, dict):
            self.testset_names, datasets = list(zip(*datasets.items()))
        trainer = pl.Trainer(gpus=[0])
        return trainer.test(self, test_dataloaders=datasets)

    @property
    def p_dropout(self):
        return self.model.p_dropout

    @p_dropout.setter
    def p_dropout(self, p):
        self.model.p_dropout = p
