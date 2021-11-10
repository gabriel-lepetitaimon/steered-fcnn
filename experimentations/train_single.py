import os
import os.path as P
from typing import Dict
from json import load
import argparse
import orion.client
import orion.storage
from orion.client import get_experiment
from orion.core.utils.exceptions import NoConfigurationError
from tempfile import NamedTemporaryFile

from src.config import parse_config


def main():
    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='specify which config file to use for the experiment')
    parser.add_argument('--gpus', help='specify which gpu should be used')
    parser.add_argument('--debug', action='store_true', help='When set, orion is run in debug mode and the experiment name is overiden by DEBUG_RUNS')
    args = parser.parse_args()

    env = {}
    if args.gpus is not None:
        env['TRIAL_GPUS'] = str(args.gpus)
    if args.debug:
        env['TRIAL_DEBUG'] = str(args.debug)
    run_experiment(args.cfg, env=env)


def run_experiment(cfg_path, env=None):
    if env is None:
        env = {}
    DEBUG = bool(env.get('TRIAL_DEBUG', False))

    # --- Parse Config ---
    cfg = parse_config(cfg_path)
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
                    cfg['trial']['id'] = len(orion_exp.fetch_trials())
                    cfg['trial']['version'] = orion_exp.version

        print('')
        print(f' === Running {orion_exp_name} ({cfg_path}): trial {cfg["trial"]["id"]} ===')
        r = run_orion(cfg_path, cfg, env)

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


def run_orion(cfg_path: str, cfg: Dict, env: Dict):
    trial_id = cfg['trial']['id']
    orion_exp_name = cfg['trial']['name']

    # --- Set Env Variable ---
    os.environ['TRIAL_ID'] = str(trial_id)
    os.environ['TRIAL_CFG_PATH'] = cfg_path
    for k, v in env.items():
        os.environ[k] = str(v)

    # --- Prepare tmp folder ---
    tmp_path = cfg['script-arguments']['tmp-dir']
    if not P.exists(tmp_path):
        os.makedirs(tmp_path)
    with NamedTemporaryFile('w+', dir=tmp_path, suffix='.yaml') as orion_cfg:
        # --- Save extended cfg file to tmp ---
        cfg['orion'].to_yaml(orion_cfg)
        orion_cfg_filepath = P.join(tmp_path, orion_cfg.name)

        # --- Prepare orion command ---
        orion_opt = " "
        exp_opt = " "
        if bool(env.get('TRIAL_DEBUG', False)):
            orion_opt += "--debug "
            exp_opt += "--exp-max-trials 1 "
        orion_cmd = (f'orion{orion_opt}hunt -c "{orion_cfg_filepath}" -n "{orion_exp_name}"{exp_opt}'
                     f'python3 run_train.py --config "{cfg_path}"')

        # --- Run orion command ---
        print('>> ', orion_cmd)
        os.system(orion_cmd)

        # --- Fetch and return run results ---
        tmp_json = P.join(tmp_path, f'{orion_exp_name}-{trial_id}.json')
        try:
            with open(tmp_json, 'r') as f:
                r = load(f)
            os.remove(tmp_json)
            return r
        except OSError:
            return {'r': -1}


if __name__ == '__main__':
    main()
