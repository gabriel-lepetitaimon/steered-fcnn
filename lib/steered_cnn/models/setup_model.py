from .steered import SteeredUNet
from .backbones import UNet


def setup_model(cfg):
    if cfg.check('backbone', 'unet'):
        args = cfg.subset('n_in,n_out,nfeatures_base,'
                          'nscale,depth,kernel,padding,'
                          'batchnorm,downsample,upsample')
        if cfg.get('steered', default=False):
            net = SteeredUNet(normalize_steer=cfg.get('normalized', True),
                              attention_mode=cfg.get('normalized', 'shared'),
                              attention_base=cfg.get('attention_base', False), **args)
        else:
            net = UNet(**args)
    return net
