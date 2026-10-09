import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 task columns A-J, found: {task_cols}')
worker_id_col = 'Task Time Required'
worker_rows = df[worker_id_col].str.strip().str.casefold().str.startswith('worker')
df_workers = df[worker_rows].copy()
workers = df_workers[worker_id_col].tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found: {workers}')
tasks = task_cols
for t in tasks:
    df_workers[t] = df_workers[t].astype(float)
cost = {}
for (idx, w) in enumerate(workers):
    for t in tasks:
        val = df_workers.iloc[idx][t]
        if pd.isnull(val) or val == '':
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        cost[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name=f'assign_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'worker_task_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'link_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    for t in tasks:
        for w in workers:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t}: assigned to {w} (time: {cost[w, t]})')
    print('--- Selected Workers ---')
    for w in workers:
        if y_vars[w].X > 0.5:
            print(f'{w}')
else:
    print(f'No optimal solution found. Status: {m.status}')