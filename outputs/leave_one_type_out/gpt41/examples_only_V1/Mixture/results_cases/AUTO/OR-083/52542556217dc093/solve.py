import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
worker_id_col = 'Task Time Required'
worker_rows = df[~df[worker_id_col].astype(str).str.strip().str.casefold().eq('worker')].copy()
worker_ids = worker_rows[worker_id_col].astype(str).tolist()
task_cols = [col for col in df.columns if col in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f"Expected 10 task columns 'A'-'J', found: {task_cols}")
time_required = {}
for idx, row in worker_rows.iterrows():
    w = str(row[worker_id_col])
    time_required[w] = {}
    for t in task_cols:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        time_required[w][t] = float(val)
model = gp.Model('WorkerTaskAssignment')
x = model.addVars(worker_ids, task_cols, vtype=gp.GRB.BINARY, name='')
y = model.addVars(worker_ids, vtype=gp.GRB.BINARY, name='')
model.setObjective(gp.quicksum((time_required[w][t] * x[w, t] for w in worker_ids for t in task_cols)), gp.GRB.MINIMIZE)
for t in task_cols:
    model.addConstr(gp.quicksum((x[w, t] for w in worker_ids)) == 1)
for w in worker_ids:
    model.addConstr(gp.quicksum((x[w, t] for t in task_cols)) <= y[w])
model.addConstr(gp.quicksum((y[w] for w in worker_ids)) == 10)
for w in worker_ids:
    for t in task_cols:
        model.addConstr(x[w, t] <= y[w])
model.optimize()
if model.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {model.objVal:.2f}')
    print('Assignment plan (worker -> task):')
    for w in worker_ids:
        for t in task_cols:
            if x[w, t].X > 0.5:
                print(f'  Worker {w} assigned to Task {t} (time: {time_required[w][t]})')
    print('Selected workers:')
    for w in worker_ids:
        if y[w].X > 0.5:
            print(f'  Worker {w}')
else:
    print(f'No optimal solution found. Status: {model.status}')