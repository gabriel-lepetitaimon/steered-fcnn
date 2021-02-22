from run_train import parse_config
import os
import argparse
from orion.storage.base import setup_storage
from orion.client import get_experiment, create_experiment
from orion.core.utils.exceptions import NoConfigurationError
from tempfile import NamedTemporaryFile


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
    if(not bool(env.get('TRIAL_DEBUG', False))):
        setup_storage(cfg.orion.storage.to_dict())
        try:
            orion_exp = get_experiment(orion_exp_name)
        except NoConfigurationError:
            n_trial = 0
        else:
            n_trial = len(orion_exp.fetch_trials())
            if n_trial == cfg.orion.experiment.max_trials:
                return False
    else:
        n_trial = 0

    # --- Set Env Variable ---
    os.environ['TRIAL_ID'] = str(n_trial)
    for k, v in env.items():
        os.environ[k] = str(v)

    # --- Launch Orion ---
    print(f'Running {orion_exp_name} ({cfg_path}): trial {n_trial}')
    tmp_path = cfg['script-arguments']['tmp-dir']
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    with NamedTemporaryFile('w+', dir=tmp_path, suffix='.yaml') as orion_cfg:
        cfg.orion.to_yaml(orion_cfg)
        orion_cfg_filepath = os.path.join(tmp_path, orion_cfg.name)

        orion_opt = " "
        if bool(env.get('TRIAL_DEBUG', False)):
            orion_opt += "--debug "
        print(f'orion{orion_opt}hunt -c "{orion_cfg_filepath}" -n "{orion_exp_name}" '
                  f'python3 run_train.py --config "{cfg_path}"')
        os.system(f'orion{orion_opt}hunt -c "{orion_cfg_filepath}" -n "{orion_exp_name}" '
                  f'python3 run_train.py --config "{cfg_path}"')

    return True


if __name__ == '__main__':
    main()
