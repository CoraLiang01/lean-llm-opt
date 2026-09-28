import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
worker_col = 'Task Time Required'
task_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
worker_rows = df[df[worker_col].astype(str).str.strip().str.casefold() != 'worker'].copy()
worker_ids = worker_rows[worker_col].astype(str).str.strip().tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
task_ids = task_cols
if len(task_ids) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(task_ids)}: {task_ids}')
cost = {}
for idx, row in worker_rows.iterrows():
    w = str(row[worker_col]).strip()
    for t in task_ids:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        cost[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x = m.addVars(worker_ids, task_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in worker_ids for t in task_ids)), gp.GRB.MINIMIZE)
for t in task_ids:
    m.addConstr(gp.quicksum((x[w, t] for w in worker_ids)) == 1, name=f'task_{t}')
for w in worker_ids:
    m.addConstr(gp.quicksum((x[w, t] for t in task_ids)) <= 1, name=f'worker_{w}')
m.addConstr(gp.quicksum((x[w, t] for w in worker_ids for t in task_ids)) == 10, name='total_assignments')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    assigned_workers = set()
    for t in task_ids:
        for w in worker_ids:
            if x[w, t].X > 0.5:
                print(f'Task {t}: Worker {w} (Time: {cost[w, t]})')
                assigned_workers.add(w)
    print(f'Assigned workers: {sorted(assigned_workers)}')
    print(f'Unassigned workers: {sorted(set(worker_ids) - assigned_workers)}')
else:
    print(f'No optimal solution found. Status: {m.status}')