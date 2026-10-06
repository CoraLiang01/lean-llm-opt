import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_col = df.columns[0]
worker_cols = list(df.columns[1:])
tasks = df[task_col].tolist()
workers = worker_cols
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)} in CSV.')
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)} in CSV.')
time_required = {}
for t in tasks:
    row = df[df[task_col] == t]
    if row.empty:
        raise ValueError(f"Task '{t}' not found in CSV.")
    for w in workers:
        val = row[w].values[0]
        if pd.isnull(val):
            raise ValueError(f"Missing time for worker '{w}', task '{t}'.")
        time_required[w, t] = float(val)

def solve_assignment_problem(tasks, workers, time_required):
    m = gp.Model('WorkerTaskAssignment')
    x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((time_required[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[w, t] for w in workers)) == 1 for t in tasks), name='')
    m.addConstrs((gp.quicksum((x[w, t] for t in tasks)) <= y[w] for w in workers), name='')
    m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_assignment_problem(tasks, workers, time_required)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')