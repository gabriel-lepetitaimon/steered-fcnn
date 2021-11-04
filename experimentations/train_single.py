import os
import os.path as P
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
    parser.add_argument('--debug', help='When set, orion is run in debug mode and the experiment name is overiden by DEBUG_RUNS')
    args = parser.parse_args()

    env = {}
    if args.gpus is not None:
        env['TRIAL_GPUS'] = args.gpus
    if args.debug is not None:
        env['TRIAL_DEBUG'] = args.debug
    run_experiment(args.cfg)


def run_experiment(cfg_path, env=None):
    if env is None:
        env = {}

    # --- Parse Config ---
    cfg = parse_config(cfg_path)
    exp_cfg = cfg.experiment
    orion_exp_name = f"{exp_cfg.name}-{exp_cfg['sub-experiment']}-{exp_cfg['sub-experiment-id']:03}"

    # --- Fetch Orion Infos ---
    if not bool(env.get('TRIAL_DEBUG', False)):
        orion.storage.base.setup_storage(cfg.orion.storage.to_dict())
        try:
            orion_exp = get_experiment(orion_exp_name)
        except NoConfigurationError:
            n_trials = 0
        else:
            n_trials = len(orion_exp.fetch_trials())
            if n_trials == cfg.orion.experiment.max_trials:
                return False
    else:
        n_trials = 0

    # --- Set Env Variable ---
    os.environ['TRIAL_ID'] = str(n_trials)
    os.environ['TRIAL_CFG_PATH'] = cfg_path
    for k, v in env.items():
        os.environ[k] = str(v)

    # --- Launch Orion ---
    print(f'Running {orion_exp_name} ({cfg_path}): trial {n_trials}')
    tmp_path = cfg['script-arguments']['tmp-dir']
    if not P.exists(tmp_path):
        os.makedirs(tmp_path)
    with NamedTemporaryFile('w+', dir=tmp_path, suffix='.yaml') as orion_cfg:
        cfg.orion.to_yaml(orion_cfg)
        orion_cfg_filepath = P.join(tmp_path, orion_cfg.name)

        orion_opt = " "
        exp_opt = " "
        if bool(env.get('TRIAL_DEBUG', False)):
            orion_opt += "--debug "
            exp_opt += "--exp-max-trials 1 "
        print(f'orion{orion_opt}hunt -c "{orion_cfg_filepath}" -n "{orion_exp_name}"{exp_opt}'
                  f'python3 run_train.py --config "{cfg_path}"')
        os.system(f'orion{orion_opt}hunt -c "{orion_cfg_filepath}" -n "{orion_exp_name}"{exp_opt}'
                      f'python3 run_train.py --config "{cfg_path}"')

        if not bool(env.get('TRIAL_DEBUG', False)):
            tmp_json = P.join(tmp_path, f'{orion_exp_name}-{n_trials}.json')
            try:
                with open(tmp_json, 'r') as f:
                    json = load(f)
                    r = json.get('rcode', -1)
                print(f'r_code={r}')
                os.remove(tmp_json)
            except OSError:
                r = -1
            print('')
            print('-'*30)
            print('')

            return 10 <= r <= 20
        else:
            return 0


if __name__ == '__main__':
    main()
