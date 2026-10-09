import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
worker_rows = df[~df['Task Time Required'].str.casefold().str.strip().eq('worker')].copy()
workers = worker_rows['Task Time Required'].tolist()
tasks = task_cols.copy()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)}: {tasks}')
time_required = {}
for (idx, row) in worker_rows.iterrows():
    w = row['Task Time Required']
    for t in tasks:
        val = row[t]
        if val == '':
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        try:
            time_required[w, t] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid time value for worker {w}, task {t}: {val}') from e
m = gp.Model('WorkerTaskAssignment')
x_keys = [(w, t) for w in workers for t in tasks]
x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time_required[w, t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name=f'task_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'worker_{w}_assign')
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x_vars[w, t] <= y_vars[w], name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (w, t) in x_keys:
        print(f'{x_vars[w, t].VarName} {x_vars[w, t].X}')
    for w in workers:
        print(f'{y_vars[w].VarName} {y_vars[w].X}')
else:
    print(f'Solver status: {m.status}')