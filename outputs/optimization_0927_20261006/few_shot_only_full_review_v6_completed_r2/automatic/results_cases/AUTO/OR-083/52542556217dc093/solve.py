import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Task Time Required' not in df.columns:
    raise KeyError("Expected 'Task Time Required' as the first column in 15.csv")
workers = df['Task Time Required'].tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)} in 15.csv')
task_columns = [col for col in df.columns if col != 'Task Time Required']
if len(task_columns) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(task_columns)} in 15.csv')

def norm(x):
    return str(x).strip()
workers = [norm(w) for w in workers]
tasks = [norm(t) for t in task_columns]
time = {}
for (idx, row) in df.iterrows():
    worker = norm(row['Task Time Required'])
    for task in tasks:
        val = row[task]
        try:
            time[worker, task] = float(val)
        except Exception as e:
            raise ValueError(f"Invalid time value for worker '{worker}', task '{task}': '{val}'") from e
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time[w, t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name=f'assign_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'worker_select_{w}')
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= 1, name=f'worker_one_task_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment ---')
    for t in tasks:
        for w in workers:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t}: assigned to Worker {w} (time: {time[w, t]})')
    print('--- Selected Workers ---')
    for w in workers:
        if y_vars[w].X > 0.5:
            print(f'Worker {w} selected')
else:
    print(f'No optimal solution found. Status: {m.status}')