import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from typing import Union, List

import sources


cfg = sources.default_config()
mlclient = MlflowClient(cfg.mlflow.uri)

psql = 'postgresql://mlflow_user:mlflow@localhost/mlflow_db'

def parse_runs(runs):
    df_metrics = pd.DataFrame()
    df_params = pd.DataFrame()
    df_miscs = pd.DataFrame()
    df_runs = pd.DataFrame()
    
    for i, r in enumerate(runs):
        df_runs.at[i, 'runID'] = r.run_id
        df_runs.at[i, 'exp'] = r.data.tags.get('exp', '-')
        df_runs.at[i, 'sub'] = r.data.tags.get('sub', '-')
        df_runs.at[i, 'subexp'] = r.data.tags.get('subexp', '-')
        
        for k, v in r.data.metrics.items():
            df_metrics.at[i, k] = v
            
        for k, v in r.data.params.items():
            df_params.at[i, ('params', k)] = v
            
        artifacts_path = r.info.artifact_uri.split('polymtl.ca')[1]
        try:
            with open(os.path.join(artifacts_path, 'misc.json'), 'r') as json_f:
                misc = AttributeDict.from_dict(json.load(json_f), recursive=True)
        except:
            pass
        else:
            for k in misc.walk():
                df_miscs.at[i, ('misc', k)] = misc[k]
    dataframes = {'run': df_runs, 'metrics': df_metrics, 'params': df_params, 'miscs': df_miscs}
    return pd.concat(dataframes.values(), axis=1, keys=dataframes.keys()).fillna('-')


def fetch_all_runs(exp):
    if isinstance(exp, str):
        exp = mlclient.get_experiment_by_name(EXP)
    import psycopg2
    conn = psycopg2.connect('postgresql://mlflow_user:mlflow@localhost/mlflow_db')
    cur = conn.cursor()
    
    run_where = f"experiment_id={exp.experiment_id} and status='FINISHED' and lifecycle_stage='active'"
    run_uuid = f"(select run_uuid from runs where {run_where})"
    
    # --- Fetch Runs ---
    cur.execute(f"select run_uuid, start_time, end_time from runs where {run_where}")
    runs = cur.fetchall()
    runs = pd.DataFrame(runs, columns=['runID', 'start_time', 'end_time'])
    runs['start_time'] = pd.to_datetime(runs['start_time'], unit='ms')
    runs['end_time'] = pd.to_datetime(runs['end_time'], unit='ms')
    runs.index = runs['runID']
    del runs['runID']

    # --- Fetch Tags ---
    cur.execute(f"select key, value, run_uuid from tags where run_uuid in {run_uuid}")
    tags = cur.fetchall()
    tags = pd.DataFrame(tags, columns=['key', 'value', 'runID'])
    tags = tags.pivot(index='runID', columns='key', values='value')
    
    # --- Fetch Params ---
    cur.execute(f"select key, value, run_uuid from params where run_uuid in {run_uuid}")
    params = cur.fetchall()
    params = pd.DataFrame(params, columns=['key', 'value', 'runID'])
    params = params.pivot(index='runID', columns='key', values='value')
    
    # --- Fetch Metrics ---
    cur.execute(f"""
    SELECT DISTINCT ON (key, run_uuid)
           key, value, run_uuid
    FROM   metrics
    WHERE run_uuid in {run_uuid}
    ORDER  BY run_uuid, key, timestamp DESC;
    """)
    metrics = cur.fetchall()
    metrics = pd.DataFrame(metrics, columns=['key', 'value', 'runID'])
    metrics = metrics.pivot(index='runID', columns='key', values='value')
    
    return pd.concat({'run': runs, 'tags':tags, 'metrics': metrics, 'params': params}, axis=1)

def fetch_runs(exp, filter=None, parse=True, psql=False):
    if isinstance(exp, str):
        exp = mlclient.get_experiment_by_name(exp)
    runs = []
    page_token = None
    while not runs or page_token is not None:
        r = mlclient.list_run_infos(exp.experiment_id, ViewType.ACTIVE_ONLY, page_token=page_token)
        page_token = r.token
        if filter is not None:
            runs += [mlclient.get_run(r.run_id) for r in runs if r.status=='FINISHED' and filter(r)]
        else:
            runs += [mlclient.get_run(r.run_id) for r in runs if r.status=='FINISHED']
    
    return parse_runs(runs) if parse else runs


def fetch_run(run_id):
    run = mlclient.get_run(run_id)
    return parse_runs([run]).iloc[0]


def fetch_metrics_history(runs: str, metrics: Union[str, List[str]]):
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = metrics
    
    df_metrics = None
    for metric in metrics:
        history = mlclient.get_metric_history(runs, metric)
        df = pd.DataFrame([{'step': _.step, metric: _.value, 
                            'timestamp': pd.Timestamp(_.timestamp, unit='ms'),
                           } for _ in history])#.drop_duplicates()
        if df_metrics is None:
            df_metrics = df
        else:
            new_step = ~df['step'].isin(df_metrics['step'])
            df_metrics = pd.merge(df_metrics, df[['step', metric]], on='step')
            missing_step = df_metrics['step'].isin(df['step'][new_step])
            df_metrics.loc[missing_step]['timestamp'] = df.loc[new_step]['timestamp']
            
    epoch = mlclient.get_metric_history(runs, 'epoch')
    df_epoch = pd.DataFrame([{'step': _.step, 'epoch': _.value} for _ in epoch]).drop_duplicates()
    df_metrics = pd.merge(df_metrics, df_epoch, on='step')
    
    return df_metrics.sort_values('step')