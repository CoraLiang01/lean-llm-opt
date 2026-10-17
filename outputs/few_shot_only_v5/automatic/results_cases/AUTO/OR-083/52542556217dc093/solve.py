import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df_columns = [col.strip() for col in df.columns]
worker_col = None
for col in df_columns:
    if col.strip().casefold() == 'worker':
        worker_col = col
        break
if worker_col is None:
    raise KeyError("Could not find 'Worker' column in CSV.")
worker_col_idx = df_columns.index(worker_col)
task_cols = df_columns[worker_col_idx + 1:]
task_cols = [col for col in task_cols if col.strip() != '']
worker_ids = df[worker_col].apply(lambda x: str(x).strip()).tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}.')
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(task_cols)}.')
time_required = {}
for idx, row in df.iterrows():
    w = str(row[worker_col]).strip()
    for t in task_cols:
        t_id = t.strip()
        val = row[t]
        try:
            time = int(str(val).strip())
        except Exception:
            raise ValueError(f"Invalid time value for worker {w}, task {t_id}: '{val}'")
        time_required[w, t_id] = time
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(worker_ids, task_cols, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(worker_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time_required[w, t] * x_vars[w, t] for w in worker_ids for t in task_cols)), gp.GRB.MINIMIZE)
for t in task_cols:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in worker_ids)) == 1, name=f'assign_{t}')
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in task_cols)) <= y_vars[w], name=f'worker_task_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in worker_ids)) == 10, name='select_10_workers')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total working hours)')
    print('\n--- Assignment Plan ---')
    assigned_workers = []
    for t in task_cols:
        for w in worker_ids:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t}: Worker {w} (Time: {time_required[w, t]})')
                assigned_workers.append(w)
                break
    print('\n--- Selected Workers ---')
    for w in worker_ids:
        if y_vars[w].X > 0.5:
            print(f'Worker {w}: SELECTED')
else:
    print(f'No optimal solution found. Status: {m.status}')