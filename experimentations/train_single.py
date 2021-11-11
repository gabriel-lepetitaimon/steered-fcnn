import os
import os.path as P
from typing import Dict
from json import load
import argparse
import orion.client
import orion.storage
from orion.client import get_experiment
from orion.core.utils.exceptions import NoConfigurationError
from tempfile import TemporaryDirectory

from src.config import parse_arguments, set_env_var


def main():
    # --- Parser ---
    cfg = parse_arguments()
    run_experiment(cfg)


def run_experiment(cfg):
    script_args = cfg['script-argument']
    DEBUG = script_args.debug
    cfg_path = script_args.config

    # --- Parse Config ---
    exp_cfg = cfg.experiment
    orion_exp_name = f"{exp_cfg.name}-{exp_cfg['sub-experiment']}-{exp_cfg['sub-experiment-id']:03}"

    if not DEBUG:
        orion.storage.base.setup_storage(cfg.orion.storage.to_dict())

    ended = False
    while not ended:
        cfg['trial'] = dict(id=0, name=orion_exp_name, version=0)
        # --- Fetch Orion Infos ---
        if not DEBUG:
            try:
                orion_exp = get_experiment(orion_exp_name)
            except NoConfigurationError:
                pass
            else:
                if orion_exp.is_done:
                    return True
                elif orion_exp.is_broken:
                    return False
                else:
                    cfg['trial']['ID'] = len(orion_exp.fetch_trials())
                    cfg['trial']['version'] = orion_exp.version

        print('')
        print(f' === Running {orion_exp_name} ({cfg_path}): trial {cfg.trial.id} ===')
        r = run_orion(cfg)

        if not DEBUG:
            if 10 <= r['r_code'] <= 20:
                print('')
                print('-'*30)
                print('')
                continue
            else:
                return False
        else:
            return True


def run_orion(cfg: Dict):
    orion_exp_name = cfg['trial']['name']
    cfg_path = cfg['script-arguments']['config']
    DEBUG = cfg['script-arguments']['debug']

    # --- Prepare tmp folder ---
    tmp_path = cfg['script-arguments']['tmp-dir']
    if not P.exists(tmp_path):
        os.makedirs(tmp_path)
    with TemporaryDirectory(dir=tmp_path, prefix=f"{orion_exp_name}-{cfg['trial']['ID']}") as tmp_dir:
        cfg['script-arguments']['tmp-dir'] = tmp_dir.name

        # --- Save orion cfg file to tmp ---
        with open(P.join(tmp_dir.name, '.orion_cfg.yaml'), 'w+') as orion_cfg:
            cfg['orion'].to_yaml(orion_cfg)
            orion_cfg_filepath = P.join(tmp_path, orion_cfg.name)

        # --- Set Env Variable ---
        set_env_var(cfg)

        # --- Prepare orion command ---
        orion_opt = " "
        exp_opt = " "
        if DEBUG:
            orion_opt += "--debug "
            exp_opt += "--exp-max-trials 1 "
        orion_cmd = (f'orion{orion_opt}hunt -c "{orion_cfg_filepath}" -n "{orion_exp_name}"{exp_opt}'
                     f'python3 run_train.py --config "{cfg_path}"')

        # --- Run orion command ---
        print('>> ', orion_cmd)
        os.system(orion_cmd)

        # --- Fetch and return run results ---
        tmp_json = P.join(tmp_path, f'result.json')
        try:
            with open(tmp_json, 'r') as f:
                r = load(f)
            os.remove(tmp_json)
            return r
        except OSError:
            return {'r': -1}


if __name__ == '__main__':
    main()
