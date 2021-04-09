from train_single import run_experiment
import argparse
import os
import os.path as P


def main():
    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', help='specify in which directory should the config files be read')
    parser.add_argument('--gpus', help='specify which gpu should be used')
    parser.add_argument('--debug', action='store_true',
                        help='When set, orion is run in debug mode and the experiment name is overridden by DEBUG_RUNS')
    args = parser.parse_args()

    train_multiple(args.cfg_path, debug=args.debug, gpus=args.gpus)


def train_multiple(path=None, debug=False, gpus=None, env=None):
    if path is None:
        path = P.abspath(P.join(P.dirname(__file__), 'EXP/'))
    if env is None:
        env = {}

    if gpus is not None:
        env['TRIAL_GPUS'] = str(gpus)
    if debug:
        env['TRIAL_DEBUG'] = str(debug)

    ended = False
    while not ended:
        cfgs = sorted(_ for _ in os.listdir(path) if _.endswith('.yaml'))
        ended = True
        for cfg in cfgs:
            if run_experiment(os.path.join(path, cfg), env):
                ended = bool(env.get('TRIAL_DEBUG', False))   # True if debug else False
                break


if __name__ == '__main__':
    main()
