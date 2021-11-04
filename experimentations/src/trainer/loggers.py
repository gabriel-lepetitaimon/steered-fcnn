import mlflow
import tempfile
import shutil
from os.path import join
import os
from mlflow.tracking import MlflowClient

from ..config import default_config

class Logs:
    def __init__(self):
        self.tmp = None

    @property
    def tmp_path(self):
        return self.tmp.name

    def setup_log(self, cfg=None):
        if cfg is None:
            cfg = default_config()

        # --- SETUP MLFOW ---
        mlflow.set_tracking_uri(cfg['mlflow']['uri'])
        mlflow.set_experiment(cfg['experiment']['name'] if not cfg['script-arguments'].debug else 'DEBUG_RUNS')
        mlflow.pytorch.autolog(log_models=False)
        tags = cfg.experiment.tags.to_dict()
        tags['subexp'] = cfg.experiment['sub-experiment']
        tags['subexpID'] = str(cfg.experiment['sub-experiment-id'])
        run_name = f"{cfg.experiment['sub-experiment']}{cfg.experiment['sub-experiment-id']}-{cfg.trial.id:02}"
        mlflow.start_run(run_name=run_name, tags=tags)

        # --- CREATE TMP ---
        os.makedirs(os.path.dirname(cfg['script-arguments']['tmp-dir']), exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=cfg['script-arguments']['tmp-dir'])
        self.tmp = tmp

        # --- SAVE CFG ---
        shutil.copy(cfg['script-arguments'].config, join(tmp.name, 'cfg.yaml'))
        mlflow.log_artifact(join(tmp.name, 'cfg.yaml'))
        # Sanity check of artifact saving
        client = MlflowClient()
        artifacts = client.list_artifacts(mlflow.active_run().info.run_id)
        if len(artifacts) != 1 or artifacts[0].path != 'cfg.yaml':
            raise RuntimeError('The sanity check for storing artifacts failed.'
                               'Interrupting the script before the training starts.')

        exp_cfg_path = os.getenv('TRIAL_CFG_PATH', None)
        if exp_cfg_path is not None:
            shutil.copy(exp_cfg_path, join(tmp.name, 'cfg_original.yaml'))
            mlflow.log_artifact(join(tmp.name, 'cfg_original.yaml'))

        with open(join(tmp.name, 'cfg_extended.yaml'), 'w') as f:
            cfg.to_yaml(f)

        mlflow.log_param('sub-experiment', cfg.experiment['sub-experiment'])
        if cfg.experiment['sub-experiment-id']:
            mlflow.log_param('sub-experiment-id', cfg.experiment['sub-experiment-id'])
        for k, v in cfg.trial.items():
            mlflow.log_param('trial.' + k, v)

        for k, v in cfg['model'].items():
            mlflow.log_param(f'model.{k}', v)
        for k, v in cfg['data-augmentation'].items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    mlflow.log_param(f'DA.{k} {k1}', v1)
            else:
                mlflow.log_param(f'DA.{k}', v)
        mlflow.log_param('dropout', cfg['hyper-parameters']['drop-out'])
        mlflow.log_param('training.file', cfg.training['dataset-file'])
        mlflow.log_param('training.dataset', cfg.training['training-dataset'])

    def save_cleanup(self):
        mlflow.log_artifacts(self.tmp.name)
        mlflow.end_run()
        self.tmp.cleanup()
