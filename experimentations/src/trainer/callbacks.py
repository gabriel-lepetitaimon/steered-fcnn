__all__ = ['ExportSegmentation', 'ExportClassification']

from typing import Any
from pytorch_lightning.callbacks import Callback
import numpy as np
from pytorch_lightning import LightningModule
import torch
from pytorch_lightning import Trainer


class ExportSegmentation(Callback):
    def __init__(self, color_map, path, dataset_names):
        super(ExportSegmentation, self).__init__()

        from .lut import prepare_lut
        self.color_lut = prepare_lut(color_map, source_dtype=np.int)
        self.dataset_names = dataset_names
        self.path = path

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.export_batch(batch, outputs, batch_idx, dataloader_idx, prefix='val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.export_batch(batch, outputs, batch_idx, dataloader_idx, prefix='test')

    def export_batch(self, batch, outputs, batch_idx, dataloader_idx, prefix):
        import os
        import cv2
        import torch
        from steered_cnn.utils import clip_pad_center

        if batch_idx:
            return

        x = batch['x']
        y = (batch['y'] != 0).float()
        y_pred = outputs.detach() > .5
        y = clip_pad_center(y, y_pred.shape)

        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_pred.shape)
            y[mask==0] = float('nan')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        diff = torch.stack((y, y_pred), dim=1)
        diff = diff.cpu().numpy()
        for i, diff_img in enumerate(diff):
            diff_img = (self.color_lut(diff_img).transpose(1, 2, 0) * 255).astype(np.uint8)
            path = os.path.abspath(os.path.join(self.path, f'{prefix}-{self.dataset_names[dataloader_idx]}-{i}.png'))
            cv2.imwrite(path, diff_img)

            
class ExportClassification(Callback):
    def __init__(self, path, n=5):
        super(ExportClassification, self).__init__()
        self.n = n
        self.path = path
        self.store = {'TP': [], 'TN': [], 'FP': [], 'FN': []}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.store_result(batch, outputs)

    def on_validation_end(self, trainer, pl_module):
        self.export_result()
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.store_result(batch, outputs)
        
    def on_test_end(self, trainer, pl_module):
        self.export_result()
        
    def store_result(self, batch, outputs):
        for i, (y, pred) in enumerate(zip(batch['y'], outputs)):
            if y==pred:
                l = self.store['TP'] if pred==1 else self.store['TN']
            else:
                l = self.store['FP'] if pred==1 else self.store['FN']
            if len(l) < self.n:
                l.append(batch['x'][i].cpu().numpy())
            elif torch.rand(1) < .1:
                l[torch.randint(size=(1,), high=self.n).item()] = batch['x'][i].cpu().numpy()
                
    def export_result(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(4, 1)
        for imgs in self.store.values():
            if imgs:
                c,h,w = imgs[0].shape
                dtype = imgs[0].dtype
                break
        else:
            return
        
        for ax, (name, imgs) in zip(axs, self.store.items()):
            r = np.ones(dtype=dtype, shape=(h,w*self.n,c))
            for i, img in enumerate(imgs):
                r[:,h*i:h*(i+1)] = img[::-1].transpose((1,2,0))
            ax.imshow(r, vmin=0, vmax=1)
            ax.set_title(name, loc='left', pad='0.5')
        fig.set_size_inches((20, 25))
        fig.tight_layout()
        fig.savefig(self.path)


class CudaMonitorCallbacks(Callback):
    def __init__(self):
        super(CudaMonitorCallbacks, self).__init__()
        self.device = torch.cuda.current_device()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch: Any, batch_idx: int, unused=0,):
        torch.cuda.utilization()
