import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if re.fullmatch('[A-J]', col)]
if len(task_cols) != 10:
    raise ValueError(f"Expected 10 task columns labeled 'A' to 'J', found: {task_cols}")
tasks = task_cols
if df.shape[0] < 12:
    raise ValueError(f'Expected at least 12 worker rows, found {df.shape[0]}')
workers = list(range(1, 13))
time = {}
for (idx, row) in df.iterrows():
    if idx >= 12:
        break
    worker = idx + 1
    time[worker] = {}
    for t in tasks:
        val = row[t]
        try:
            time[worker][t] = float(val)
        except Exception:
            raise ValueError(f'Invalid time value for worker {worker}, task {t}: {val}')
m = gp.Model('WorkerTaskAssignment')
x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time[w][t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'assign_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= y[w], name=f'worker_task_{w}')
m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x[w, t] <= y[w], name=f'link_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} (Total working hours)')
    print('\n--- Assignment Plan ---')
    assigned_workers = []
    for w in workers:
        if y[w].X > 0.5:
            assigned_workers.append(w)
            for t in tasks:
                if x[w, t].X > 0.5:
                    print(f'Worker {w} assigned to Task {t} (Time: {time[w][t]:.2f})')
    print(f'\nSelected workers (assigned): {assigned_workers}')
else:
    print(f'No optimal solution found. Status: {m.status}')