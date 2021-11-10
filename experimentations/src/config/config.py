__all__ = ['default_config', 'parse_config']

import os.path as P

from .attribute_dict import AttributeDict


ORION_CFG_PATH = P.join(P.dirname(P.abspath(__file__)), 'orion_config.yaml')
GLOBAL_CFG_PATH = P.join(P.dirname(P.abspath(__file__)), 'global_config.yaml')
DEFAULT_EXP_CFG_PATH = P.join(P.dirname(P.abspath(__file__)), 'default_exp_config.yaml')


def default_config():
    with open(ORION_CFG_PATH, 'r') as f:
        orion_config = AttributeDict.from_yaml(f)
    with open(GLOBAL_CFG_PATH, 'r') as f:
        global_config = AttributeDict.from_yaml(f)
        global_config['orion'] = orion_config
    with open(DEFAULT_EXP_CFG_PATH, 'r') as f:
        global_config.recursive_update(AttributeDict.from_yaml(f))
    return global_config


def parse_config(cfg_file):
    default_cfg = default_config()
    if cfg_file is None:
        return default_cfg
    with open(cfg_file, 'r') as f:
        exp_config = AttributeDict.from_yaml(f)

    # --- Preprocess cfg file
    if "sub-experiment" not in exp_config['experiment']:
        exp_config['experiment'] = f'[{exp_config.tags["exp"]}] {exp_config.tags.get("sub","")}'
    exp_config = exp_config.filter(lambda k, v: not (isinstance(v, str) and v.startswith('orion~')), recursive=True)

    return default_cfg.recursive_update(exp_config)
