from run_train import parse_config
import os
import argparse
from orion.storage.base import setup_storage
from orion.client import get_experiment, create_experiment
from orion.core.utils.exceptions import NoConfigurationError
from tempfile import TemporaryFile


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
    setup_storage(**cfg['orion']['storage'])
    try:
        orion_exp = get_experiment(orion_exp_name)
    except NoConfigurationError:
        orion_exp = create_experiment(orion_exp_name, **cfg.orion.experiment)
    n_trial = len(orion_exp.fetch_trials())
    if n_trial == cfg.orion.experiment.max_trials:
        return False

    # --- Set Env Variable ---
    os.environ['TRIAL_ID'] = str(n_trial)
    for k, v in env.items():
        os.environ[k] = v

    # --- Launch Orion ---
    with TemporaryFile() as orion_cfg:
        cfg.orion.to_yaml(orion_cfg)

        orion_opt = " "
        if env.get('TRIAL_DEBUG', False):
            orion_opt += "--debug "
        os.system(f'orion{orion_opt}hunt -c "{orion_cfg.name}" -n "{orion_exp_name}" '
                  f'python3 run_train.py --config "{cfg_path}"')

    return True


if __name__ == '__main__':
    main()
