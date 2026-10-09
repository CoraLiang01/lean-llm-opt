import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
tasks = [col.strip() for col in task_cols]
worker_rows = df[~df['Task Time Required'].str.strip().str.casefold().eq('worker')].copy()
workers = worker_rows['Task Time Required'].str.strip().tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
time_required = {}
for (idx, row) in worker_rows.iterrows():
    w = row['Task Time Required'].strip()
    for t in tasks:
        val = row[t]
        try:
            time_required[w, t] = float(val)
        except Exception:
            raise ValueError(f"Missing or invalid time for worker {w}, task {t}: '{val}'")
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time_required[w, t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name=f'assign_task_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'worker_task_limit_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'assign_valid_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    for t in tasks:
        for w in workers:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t}: assigned to Worker {w} (time: {time_required[w, t]})')
    print('--- Selected Workers ---')
    for w in workers:
        if y_vars[w].X > 0.5:
            print(f'Worker {w} selected')
else:
    print(f'No optimal solution found. Status: {m.status}')