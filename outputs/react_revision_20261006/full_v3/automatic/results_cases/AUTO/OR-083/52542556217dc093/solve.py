import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
row_label_col = df.columns[0]
worker_caption_idx = df[df[row_label_col].astype(str).str.strip().str.casefold() == 'worker'].index
if len(worker_caption_idx) != 1:
    raise ValueError("Could not uniquely identify the 'Worker' caption row in the CSV.")
worker_caption_idx = worker_caption_idx[0]
worker_rows = df.iloc[worker_caption_idx + 1:]
workers = worker_rows[row_label_col].astype(str).tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}.')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 tasks (columns A-J), found {len(task_cols)}.')
tasks = task_cols
cost = {}
for (i, w) in enumerate(workers):
    row = worker_rows.iloc[i]
    for t in tasks:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}.')
        cost[w, t] = float(val)

def solve_assignment_problem(workers, tasks, cost):
    m = gp.Model('WorkerTaskAssignment')
    m.Params.MIPGap = 0.0001
    x = m.addVars([(w, t) for w in workers for t in tasks], vtype=gp.GRB.BINARY, name='')
    y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    for t in tasks:
        m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'task_{t}')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= y[w], name=f'worker_{w}_assign')
    m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
    for w in workers:
        for t in tasks:
            m.addConstr(x[w, t] <= y[w], name='')
    m.optimize()
    return m
m = solve_assignment_problem(workers, tasks, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')