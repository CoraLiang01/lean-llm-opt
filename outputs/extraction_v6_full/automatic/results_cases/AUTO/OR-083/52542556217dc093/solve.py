import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if re.fullmatch('[A-J]', col)]
tasks = list(task_cols)
worker_rows = df[df[task_cols].notnull().all(axis=1)].copy()
if worker_rows.shape[0] != 12:
    raise ValueError(f'Expected 12 workers (rows with all task times present), found {worker_rows.shape[0]}.')
workers = [str(i + 1) for i in range(worker_rows.shape[0])]
cost = {}
for i, w in enumerate(workers):
    for t in tasks:
        val = worker_rows.iloc[i][t]
        if pd.isnull(val):
            raise ValueError(f'Missing time for worker {w}, task {t}.')
        cost[w, t] = float(val)
m = gp.Model('WorkerTaskAssignment')
x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='x')
m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'Task_{t}_assigned')
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= 1, name=f'Worker_{w}_atmost1')
m.addConstr(gp.quicksum((x[w, t] for w in workers for t in tasks)) == 10, name='TotalAssignments10')
m.optimize()