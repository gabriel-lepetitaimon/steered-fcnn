from .steered import SteeredUNet
from .backbones import UNet


def setup_model(cfg, n_in, n_out):
    if cfg.check('backbone', 'unet'):
        args = cfg.subset('nfeatures,'
                          'nscale,depth,padding,'
                          'batchnorm,downsampling,upsampling')
        if cfg.get('steered', default=False):
            steered = cfg.get('steered')
            if isinstance(steered, str):
                steered = {'steering': steered}
            if steered.get('steering', 'attention') == 'attention':
                steered['steering'] = 'attention'
                if steered.get('attention_mode', False):
                    steered['attention_mode'] = 'shared'
            else:
                steered['attention_mode'] = False
            net = SteeredUNet(n_in, n_out,
                              rho_nonlinearity=steered.get('rho_nonlinearity', None),
                              base=steered.get('base', None),
                              attention_mode=steered.get('attention_mode'),
                              attention_base=steered.get('attention_base', False), **args)
        else:
            net = UNet(n_in, n_out, kernel=cfg.get('kernel',3), **args)
    return net
