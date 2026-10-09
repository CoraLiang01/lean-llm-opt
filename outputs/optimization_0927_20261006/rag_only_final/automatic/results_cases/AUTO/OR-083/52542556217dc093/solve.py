import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]
worker_rows = df[df['Task Time Required'].str.strip().casefold() != 'worker'].copy()
worker_ids = worker_rows['Task Time Required'].tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
task_ids = [col for col in df.columns if col != 'Task Time Required']
if len(task_ids) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(task_ids)}: {task_ids}')
time_required = {}
for (_, row) in worker_rows.iterrows():
    w = row['Task Time Required']
    for t in task_ids:
        val = row[t].strip()
        if val == '':
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        try:
            time_required[w, t] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid time value for worker {w}, task {t}: {val}') from e
m = Model()
x_vars = m.addVars(worker_ids, task_ids, vtype=GRB.BINARY, name='')
y_vars = m.addVars(worker_ids, vtype=GRB.BINARY, name='')
for t in task_ids:
    m.addConstr(quicksum((x_vars[w, t] for w in worker_ids)) == 1, name=f'assign_task_{t}')
for w in worker_ids:
    m.addConstr(quicksum((x_vars[w, t] for t in task_ids)) <= y_vars[w], name=f'worker_task_link_{w}')
m.addConstr(quicksum((y_vars[w] for w in worker_ids)) == 10, name='select_10_workers')
for w in worker_ids:
    m.addConstr(quicksum((x_vars[w, t] for t in task_ids)) <= 1, name=f'worker_one_task_{w}')
for w in worker_ids:
    for t in task_ids:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'link_{w}_{t}')
m.setObjective(quicksum((time_required[w, t] * x_vars[w, t] for w in worker_ids for t in task_ids)), GRB.MINIMIZE)
m.optimize()