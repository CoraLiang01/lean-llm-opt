import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 task columns (A-J), found: {task_cols}')
worker_id_col = 'Task Time Required'
worker_rows = df[df[worker_id_col].str.strip().str.casefold() != 'worker'].copy()
worker_ids = worker_rows[worker_id_col].tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found: {worker_ids}')
cost = {}
for (i, w) in enumerate(worker_ids):
    for t in task_cols:
        val = worker_rows.iloc[i][t]
        try:
            cost[w, t] = float(val)
        except Exception:
            raise ValueError(f"Missing or invalid time for worker {w}, task {t}: '{val}'")
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(worker_ids, task_cols, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(worker_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x_vars[w, t] for w in worker_ids for t in task_cols)), gp.GRB.MINIMIZE)
for t in task_cols:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in worker_ids)) == 1, name=f'assign_{t}')
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in task_cols)) <= y_vars[w], name=f'worker_task_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in worker_ids)) == 10, name='select_10_workers')
for w in worker_ids:
    for t in task_cols:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'assign_valid_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    for t in task_cols:
        for w in worker_ids:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t} assigned to Worker {w} (time: {cost[w, t]})')
    print('--- Selected Workers ---')
    for w in worker_ids:
        if y_vars[w].X > 0.5:
            print(f'Worker {w} selected')
else:
    print(f'No optimal solution found. Status: {m.status}')