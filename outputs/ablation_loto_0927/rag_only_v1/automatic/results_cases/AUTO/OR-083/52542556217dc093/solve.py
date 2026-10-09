import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
worker_rows = df['Task Time Required'].astype(str).str.strip()
worker_ids = worker_rows[worker_rows != 'Worker'].tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
task_ids = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_ids) != 10:
    raise ValueError(f'Expected 10 tasks (A-J), found {len(task_ids)}: {task_ids}')
worker_idx_map = {wid: idx for (idx, wid) in enumerate(worker_ids, start=1)}
cost = {}
for w in worker_ids:
    row_idx = worker_idx_map[w]
    for t in task_ids:
        val = df.at[row_idx, t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        cost[w, t] = float(val)
m = Model()
x = m.addVars(worker_ids, task_ids, vtype=GRB.BINARY, name='')
y = m.addVars(worker_ids, vtype=GRB.BINARY, name='')
m.setObjective(sum((cost[w, t] * x[w, t] for w in worker_ids for t in task_ids)), GRB.MINIMIZE)
for t in task_ids:
    m.addConstr(sum((x[w, t] for w in worker_ids)) == 1, name=f'assign_task_{t}')
for w in worker_ids:
    m.addConstr(sum((x[w, t] for t in task_ids)) <= y[w], name=f'worker_task_{w}')
m.addConstr(sum((y[w] for w in worker_ids)) == 10, name='select_10_workers')
for w in worker_ids:
    for t in task_ids:
        m.addConstr(x[w, t] <= y[w], name=f'assign_if_selected_{w}_{t}')
m.optimize()