import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 task columns (A-J), found: {task_cols}')
worker_id_col = 'Task Time Required'
worker_rows = df[~df[worker_id_col].astype(str).str.strip().str.casefold().eq('worker')].copy()
worker_ids = worker_rows[worker_id_col].astype(str).str.strip().tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found: {worker_ids}')
tasks = [col for col in task_cols]
cost = {}
for idx, row in worker_rows.iterrows():
    w = str(row[worker_id_col]).strip()
    for t in tasks:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        cost[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x = m.addVars(worker_ids, tasks, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in worker_ids for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[w, t] for w in worker_ids)) == 1, name=f'assign_task_{t}')
for w in worker_ids:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= 1, name=f'assign_worker_{w}')
m.addConstr(gp.quicksum((x[w, t] for w in worker_ids for t in tasks)) == 10, name='total_assignments')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    for w in worker_ids:
        for t in tasks:
            if x[w, t].X > 0.5:
                print(f'Worker {w} assigned to Task {t} (time: {cost[w, t]})')
else:
    print(f'No optimal solution found. Status: {m.status}')