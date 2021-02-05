import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional.classification import auroc, iou
from pytorch_lightning.metrics.functional import f1
import numpy as np
from os.path import abspath

from .utils import prepare_lut
from .model import clip_pad_center


class BinaryClassifierNet(pl.LightningModule):
    def __init__(self, model, loss='BCE', optimizer=None, lr=1e-3):
        super(BinaryClassifierNet, self).__init__()
        self.model = model

        self.accuracy = pl.metrics.Accuracy()
        self.lr = lr
        if loss == 'dice':
            from .losses import binary_dice_loss
            self.loss_f = lambda y_hat, y: binary_dice_loss(torch.sigmoid(y_hat), y)
        else:
            self.loss_f = lambda y_hat, y: F.binary_cross_entropy_with_logits(y_hat, y)
        
        if optimizer is None:
            optimizer = {'type': 'Adam'}
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = (batch['y']!=0).float()
        y_hat = self.model(x, **{k: v for k,v in batch.items() if k not in ('x','y','mask')})
        y = clip_pad_center(y, y_hat.shape)
        y_hat = y_hat.squeeze(1)

        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_hat.shape)
            y_hat = y_hat[mask != 0]
            y = y[mask != 0]
        loss = self.loss_f(y_hat, y)
        self.log('train-loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = (batch['y']!=0).float()
        y_hat = self.model(x, **{k: v for k,v in batch.items() if k not in ('x','y','mask')})
        y = clip_pad_center(y, y_hat.shape)
        y_hat = y_hat.squeeze(1)
        y_sig = torch.sigmoid(y_hat)
        y_pred = y_sig > .5

        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_hat.shape)
            y_hat = y_hat[mask != 0]
            y_sig = y_sig[mask != 0]
            y = y[mask != 0]
        
        y = y.flatten()
        y_hat = y_hat.flatten()
        y_sig = y_sig.flatten()

        self.log('valid-loss', self.loss_f(y_hat, y))
        self.log('valid-acc', self.accuracy(y_sig, y))
        self.log('valid-roc', auroc(y_sig, y))
        self.log('valid-iou', iou(y_sig>0.5, y))
        
        return y_pred

    def test_step(self, batch, batch_idx):
        x = batch['x']
        y = (batch['y']!=0).float()
        y_hat = self.model(x, **{k: v for k,v in batch.items() if k not in ('x','y','mask')})
        y = clip_pad_center(y, y_hat.shape)
        y_hat = y_hat.squeeze(1)
        y_sig = torch.sigmoid(y_hat)
        y_pred = y_sig > .5

        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_hat.shape)
            y_hat = y_hat[mask != 0]
            y_sig = y_sig[mask != 0]
            y = y[mask != 0]
            
        y = y.flatten()
        y_hat = y_hat.flatten()
        y_sig = y_sig.flatten()

        self.log('test-loss', self.loss_f(y_hat, y))
        self.log('test-acc', self.accuracy(y_pred, y))
        self.log('test-roc', auroc(y_sig, y))
        self.log('test-iou', iou(y_sig>0.5, y))
        
        return y_pred

    def configure_optimizers(self):
        opt = self.optimizer
        if opt['type'].lower() in ('adam', 'adamax', 'adamw'):
            Adam = {'adam': torch.optim.Adam,
                    'adamax': torch.optim.Adamax,
                    'adamw': torch.optim.AdamW}[opt['type'].lower()]
            kwargs = {k:v for k,v in opt.items() if k in ('weight_decay','amsgrad', 'eps')}
            optimizer = Adam(self.parameters(), lr=self.lr, betas=(opt.get('beta', .9), opt.get('beta_sqr', .999)), **kwargs)
        elif opt['type'].lower() == 'asgd':
            kwargs = {k:v for k,v in opt.items() if k in ('lambd','alpha', 't0', 'weight_decay')}
            optimizer = torch.optim.ASGD(self.parameters(), lr=self.lr, **kwargs)
        elif opt['type'].lower() == 'sgd':
            kwargs = {k:v for k,v in opt.items() if k in ('momentum','dampening', 'nesterov', 'weight_decay')}
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, **kwargs)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def __call__(*args, **kwargs):
        return self.model(*args, **kwargs)


class ExportValidation(Callback):
    def __init__(self, color_map, path):
        super(ExportValidation, self).__init__()
        self.color_lut = prepare_lut(color_map, source_dtype=np.int)
        self.path = path
        if self.path.endswith('/'):
            self.path += 'val%i.png'
        
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        import cv2
        import mlflow
        x = batch['x']
        y = (batch['y']!=0).float()
        y_pred = outputs.detach().cpu()
        y = clip_pad_center(y, y_pred.shape)
        
        diff = torch.stack((y, y_pred), dim=1)
        diff = diff.numpy()
        for i, diff_img in enumerate(diff):
            diff_img = (self.color_lut(diff_img).transpose(1, 2, 0) * 255).astype(np.uint8)
            path = abspath(self.path % i)
            cv2.imwrite(path, diff_img)
            mlflow.log_artifact(path)
