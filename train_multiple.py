from train_single import run_experiment
import argparse
import os


def main():
    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', help='specify in which directory should the config files be read', default='experiments/')
    parser.add_argument('--gpus', help='specify which gpu should be used')
    parser.add_argument('--debug', help='When set, orion is run in debug mode and the experiment name is overiden by DEBUG_RUNS')
    args = parser.parse_args()

    env = {}
    if args.gpus is not None:
        env['TRIAL_GPUS'] = args.gpus
    if args.debug is not None:
        env['TRIAL_DEBUG'] = args.debug
    run_experiment(args.cfg)


def run_multiple(path):
    ended = False
    while not ended:
        cfgs = sorted(_ for _ in os.listdir(path) if _.endswith('.yaml'))
        ended = True
        for cfg in cfgs:
            if run_experiment(cfg):
                ended = False
                break


if __name__ == '__main__':
    main()
