import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
worker_rows = df[df['Task Time Required'].str.strip().str.casefold() != 'worker']
worker_rows = worker_rows[worker_rows['Task Time Required'].str.strip() != '']
worker_ids = worker_rows['Task Time Required'].tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
task_ids = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_ids) != 10:
    raise ValueError(f'Expected 10 tasks (A-J), found {len(task_ids)}: {task_ids}')
cost = {}
for (_, row) in worker_rows.iterrows():
    w = row['Task Time Required']
    for t in task_ids:
        val = row[t]
        try:
            cost[w, t] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid or missing time value for worker {w}, task {t}: {val}')
if len(cost) != len(worker_ids) * len(task_ids):
    raise ValueError(f'Cost dictionary incomplete: expected {len(worker_ids) * len(task_ids)}, got {len(cost)}')
m = Model('worker_task_assignment')
assign_vars = m.addVars(worker_ids, task_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
select_vars = m.addVars(worker_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.addConstr(quicksum((select_vars[w] for w in worker_ids)) == 10, name='select_10_workers')
for t in task_ids:
    m.addConstr(quicksum((assign_vars[w, t] for w in worker_ids)) == 1, name=f'assign_task_{t}')
for w in worker_ids:
    m.addConstr(quicksum((assign_vars[w, t] for t in task_ids)) == select_vars[w], name=f'worker_task_{w}')
for w in worker_ids:
    for t in task_ids:
        m.addConstr(assign_vars[w, t] <= select_vars[w], name='')
m.setObjective(quicksum((cost[w, t] * assign_vars[w, t] for w in worker_ids for t in task_ids)), GRB.MINIMIZE)
m.optimize()