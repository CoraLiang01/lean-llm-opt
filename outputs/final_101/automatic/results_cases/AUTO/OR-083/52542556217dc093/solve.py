import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
tasks = [col.strip() for col in task_cols]
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks (A-J), found: {tasks}')
worker_rows = df[~df['Task Time Required'].astype(str).str.strip().str.casefold().eq('worker')]
worker_rows = worker_rows[~worker_rows['Task Time Required'].isnull()]
workers = worker_rows['Task Time Required'].astype(str).str.strip().tolist()
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found: {workers}')
cost = {}
for _, row in worker_rows.iterrows():
    w = str(row['Task Time Required']).strip()
    for t in tasks:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        cost[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
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
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('\n--- Assignment Plan ---')
    assigned_workers = []
    for w in workers:
        if y[w].X > 0.5:
            assigned_workers.append(w)
            for t in tasks:
                if x[w, t].X > 0.5:
                    print(f'Worker {w} assigned to Task {t} (time: {cost[w, t]})')
    print(f'\nSelected workers ({len(assigned_workers)}): {assigned_workers}')
else:
    print(f'No optimal solution found. Status: {m.status}')