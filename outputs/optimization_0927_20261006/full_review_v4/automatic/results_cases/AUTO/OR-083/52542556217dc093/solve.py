import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [col for col in df.columns if col.strip() not in ['Task Time Required']]
tasks = [col.strip() for col in task_cols]
num_tasks = len(tasks)
if num_tasks != 10:
    raise ValueError(f'Expected 10 tasks, found {num_tasks}: {tasks}')
worker_id_col = 'Task Time Required'
worker_ids = df[worker_id_col].tolist()
if worker_ids[0].strip().casefold() == 'worker':
    worker_ids = worker_ids[1:]
    df_data = df.iloc[1:].copy()
else:
    df_data = df.copy()
worker_ids = [w.strip() for w in worker_ids]
num_workers = len(worker_ids)
if num_workers != 12:
    raise ValueError(f'Expected 12 workers, found {num_workers}: {worker_ids}')
df_data.index = worker_ids
cost_matrix = {}
for w in worker_ids:
    for t in tasks:
        val = df_data.loc[w, t]
        try:
            cost_matrix[w, t] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid time value for worker {w}, task {t}: {val}')
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(worker_ids, tasks, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(worker_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_matrix[w, t] * x_vars[w, t] for w in worker_ids for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in worker_ids)) == 1, name=f'Task_{t}_assign')
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'Worker_{w}_assign_leq_y')
m.addConstr(gp.quicksum((y_vars[w] for w in worker_ids)) == 10, name='Select_10_workers')
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= 1, name=f'Worker_{w}_assign_leq_1')
for w in worker_ids:
    for t in tasks:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'Link_{w}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('\n--- Assignment Plan ---')
    for t in tasks:
        for w in worker_ids:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t} assigned to Worker {w} (time: {cost_matrix[w, t]})')
    print('\n--- Selected Workers ---')
    for w in worker_ids:
        if y_vars[w].X > 0.5:
            print(f'Worker {w} selected')
else:
    print(f'No optimal solution found. Status: {m.status}')