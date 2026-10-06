import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
worker_id_col = 'Task Time Required'

def is_worker_row(val):
    if isinstance(val, str):
        return not bool(re.search('worker', val, re.IGNORECASE))
    return True
worker_rows = df[df[worker_id_col].apply(is_worker_row)].copy()
workers = [str(w) for w in worker_rows[worker_id_col]]
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 tasks (columns A-J), found {len(task_cols)}: {task_cols}')
tasks = task_cols.copy()
time = {}
for idx, row in worker_rows.iterrows():
    w = str(row[worker_id_col])
    for t in tasks:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        time[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'task_{t}_assign')
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= y[w], name=f'worker_{w}_assign')
m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x[w, t] <= y[w], name=f'link_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('\n--- Assignment Plan ---')
    selected_workers = [w for w in workers if y[w].X > 0.5]
    print(f'Selected workers (total {len(selected_workers)}): {selected_workers}')
    print('\nTask assignments:')
    for t in tasks:
        for w in workers:
            if x[w, t].X > 0.5:
                print(f'  Task {t}: Worker {w} (time: {time[w, t]})')
else:
    print(f'No optimal solution found. Status: {m.status}')