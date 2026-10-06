import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
missing_cols = set(task_cols) - set(df.columns)
if missing_cols:
    raise ValueError(f'Missing required task columns in CSV: {missing_cols}')
df_workers = df.iloc[1:13].copy()
df_workers.reset_index(drop=True, inplace=True)
worker_ids = [f'Worker {i}' for i in range(1, 13)]
if len(df_workers) != 12:
    raise ValueError(f'Expected 12 worker rows, found {len(df_workers)}')
for t in task_cols:
    if df_workers[t].isnull().any():
        raise ValueError(f"Missing time value for some worker(s) in task column '{t}'")
cost = {}
for w_idx, w in enumerate(worker_ids):
    for t in task_cols:
        val = df_workers.loc[w_idx, t]
        if not np.isfinite(val):
            raise ValueError(f'Non-finite time value for {w}, task {t}')
        cost[w, t] = float(val)
W = worker_ids
T = task_cols
m = Model()
x = m.addVars(W, T, vtype=GRB.BINARY, name='')
y = m.addVars(W, vtype=GRB.BINARY, name='')
m.setObjective(sum((cost[w, t] * x[w, t] for w in W for t in T)), GRB.MINIMIZE)
for t in T:
    m.addConstr(sum((x[w, t] for w in W)) == 1, name=f'assign_task_{t}')
for w in W:
    m.addConstr(sum((x[w, t] for t in T)) <= y[w], name=f'worker_task_{w}')
m.addConstr(sum((y[w] for w in W)) == 10, name='select_10_workers')
for w in W:
    for t in T:
        m.addConstr(x[w, t] <= y[w], name=f'assign_if_selected_{w}_{t}')
m.optimize()