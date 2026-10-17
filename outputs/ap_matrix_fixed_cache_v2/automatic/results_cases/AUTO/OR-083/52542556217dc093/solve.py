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
worker_rows = df[df[worker_id_col].str.strip().str.casefold() != 'worker']
worker_ids = worker_rows[worker_id_col].tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
cost = {}
for idx, row in worker_rows.iterrows():
    w = row[worker_id_col]
    for t in task_cols:
        val = row[t]
        try:
            cost[w, t] = float(val)
        except Exception as e:
            raise ValueError(f"Invalid or missing time value for worker {w}, task {t}: '{val}'") from e
W = worker_ids
T = task_cols
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(W, T, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(W, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x_vars[w, t] for w in W for t in T)), gp.GRB.MINIMIZE)
for t in T:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in W)) == 1, name=f'assign_task_{t}')
for w in W:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in T)) <= y_vars[w], name=f'worker_task_limit_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in W)) == 10, name='select_10_workers')
for w in W:
    for t in T:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'assign_valid_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    for t in T:
        for w in W:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t} assigned to Worker {w} (time: {cost[w, t]})')
    print('--- Selected Workers ---')
    for w in W:
        if y_vars[w].X > 0.5:
            print(f'Worker {w} selected')
else:
    print(f'No optimal solution found. Status: {m.status}')