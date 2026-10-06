import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f"Expected 10 task columns 'A'-'J', found: {task_cols}")
tasks = task_cols.copy()
worker_rows = df[df['Task Time Required'].astype(str).str.strip().str.match('^\\d+$')].copy()
worker_rows['worker_id'] = worker_rows['Task Time Required'].astype(str).str.strip()
workers = worker_rows['worker_id'].tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found: {workers}')
cost = {}
for (_, row) in worker_rows.iterrows():
    w = row['worker_id']
    for t in tasks:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        cost[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x = m.addVars([(w, t) for w in workers for t in tasks], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'task_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= 1, name=f'worker_{w}')
m.addConstr(gp.quicksum((x[w, t] for w in workers for t in tasks)) == 10, name='total_assignments')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for w in workers:
        for t in tasks:
            var = x[w, t]
            if var.X > 0.5:
                print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')