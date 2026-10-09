import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 task columns (A-J), found: {task_cols}')
worker_col = None
for col in df.columns:
    if re.fullmatch('\\s*Task Time Required\\s*', col, re.IGNORECASE):
        worker_col = col
        break
if worker_col is None:
    raise ValueError("Could not find worker identifier column (expected 'Task Time Required').")
worker_rows = df[~df[worker_col].str.casefold().str.strip().eq('worker')].copy()
if worker_rows.shape[0] != 12:
    raise ValueError(f'Expected 12 workers, found {worker_rows.shape[0]} rows after filtering.')
workers = [w for w in worker_rows[worker_col]]
tasks = [t for t in task_cols]
cost = {}
for (_, row) in worker_rows.iterrows():
    w = row[worker_col]
    for t in tasks:
        val = row[t]
        try:
            cost_val = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing time value for worker {w}, task {t}: '{val}'")
        cost[w, t] = cost_val
if len(cost) != 12 * 10:
    raise ValueError(f'Cost matrix incomplete: expected 120 entries, got {len(cost)}.')
m = gp.Model('WorkerTaskAssignment')
x_keys = [(w, t) for w in workers for t in tasks]
x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name='task_assign_' + t)
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name='worker_assign_' + str(w))
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x_vars[w, t] <= y_vars[w], name='link_' + str(w) + '_' + t)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for w in workers:
        print(f'y[{w}] {y_vars[w].VarName} {y_vars[w].X}')
    for (w, t) in x_keys:
        print(f'x[{w},{t}] {x_vars[w, t].VarName} {x_vars[w, t].X}')
else:
    print(f'Solver status: {m.status}')