import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
worker_col = df.columns[0]
workers = df[worker_col].astype(str).tolist()
task_cols = [col for col in df.columns if col != worker_col]
tasks = [col for col in task_cols]
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}')
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)}')
time_required = {}
for (idx, row) in df.iterrows():
    w = str(row[worker_col])
    for t in tasks:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time for worker {w}, task {t}')
        time_required[w, t] = float(val)
if len(time_required) != len(workers) * len(tasks):
    raise ValueError('Mismatch in time_required data size.')

def solve_assignment_problem(workers, tasks, time_required):
    m = gp.Model('WorkerTaskAssignment')
    m.Params.MIPGap = 0.0001
    x = m.addVars([(w, t) for w in workers for t in tasks], vtype=gp.GRB.BINARY, name='')
    y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((time_required[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    for t in tasks:
        m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'assign_task_{t}')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= y[w], name=f'worker_task_limit_{w}')
    m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
    for w in workers:
        for t in tasks:
            m.addConstr(x[w, t] <= y[w], name=f'assign_only_if_selected_{w}_{t}')
    m.optimize()
    return m
m = solve_assignment_problem(workers, tasks, time_required)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')