import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
tasks = [col.strip() for col in task_cols]
worker_col_candidates = [col for col in df.columns if re.search('worker', col, re.IGNORECASE)]
if worker_col_candidates:
    worker_col = worker_col_candidates[0]
    df = df[df[worker_col].notnull()]
    workers = df[worker_col].astype(str).str.strip().tolist()
else:
    workers = df.index.astype(str).tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)}: {tasks}')
worker_to_row = {}
if worker_col_candidates:
    for idx, w in enumerate(workers):
        worker_to_row[w] = df.index[df[worker_col].astype(str).str.strip() == w][0]
else:
    for idx, w in enumerate(workers):
        worker_to_row[w] = idx
cost = {}
for w in workers:
    row_idx = worker_to_row[w]
    for t in tasks:
        val = df.loc[row_idx, t]
        if pd.isnull(val):
            raise ValueError(f"Missing assignment cost for worker '{w}', task '{t}'")
        cost[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'assign_task_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= 1, name=f'assign_worker_{w}')
m.addConstr(gp.quicksum((x[w, t] for w in workers for t in tasks)) == 10, name='total_assignments')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan (Worker -> Task) ---')
    for w in workers:
        for t in tasks:
            if x[w, t].X > 0.5:
                print(f'  Worker {w} assigned to Task {t} (time: {cost[w, t]})')
else:
    print(f'No optimal solution found. Status: {m.status}')