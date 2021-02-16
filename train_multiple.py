import yaml
import os
import argparse




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='specify where to look', default='experiments/')
    parser.add_argument('--gpus', help='specify which gpu should be used')
    args = parser.parse_args()

    exp = pull_experiment(args.path)


def pull_experiment(path):
    files = os.listdir(path.path)
    files = sorted([_ for _ in files if _.endswith('.yaml')])




if __name__ == '__main__':
    main()