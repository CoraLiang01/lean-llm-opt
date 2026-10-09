import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_columns = [col for col in df.columns if col.strip() not in ['Task Time Required']]
tasks = task_columns
worker_rows = df[df['Task Time Required'].str.strip().str.casefold() != 'worker']
workers = worker_rows['Task Time Required'].tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)}: {tasks}')
time_required = {}
for (_, row) in worker_rows.iterrows():
    w = row['Task Time Required']
    time_required[w] = {}
    for t in tasks:
        val = row[t]
        try:
            time_required[w][t] = float(val)
        except Exception as e:
            raise ValueError(f"Invalid time value for worker {w}, task {t}: '{val}'") from e
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time_required[w][t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name=f'assign_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'worker_task_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'link_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('\n--- Assignment Plan ---')
    selected_workers = [w for w in workers if y_vars[w].X > 0.5]
    print(f'Selected workers (assigned to tasks): {selected_workers}')
    for t in tasks:
        for w in workers:
            if x_vars[w, t].X > 0.5:
                print(f'  Task {t}: Worker {w} (time: {time_required[w][t]})')
else:
    print(f'No optimal solution found. Status: {m.status}')