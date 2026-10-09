import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_rows = df[~df['Task Time Required'].str.strip().str.casefold().eq('worker')].copy()
worker_ids = worker_rows['Task Time Required'].str.strip().tolist()
num_workers = len(worker_ids)
if num_workers != 12:
    raise ValueError(f'Expected 12 workers, found {num_workers}')
task_ids = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
num_tasks = len(task_ids)
if num_tasks != 10:
    raise ValueError(f'Expected 10 tasks, found {num_tasks}')
assignment_cost = {}
for (idx, row) in worker_rows.iterrows():
    worker = row['Task Time Required'].strip()
    for task in task_ids:
        val = row[task].strip()
        if val == '':
            raise ValueError(f'Missing time value for worker {worker}, task {task}')
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Non-numeric time value for worker {worker}, task {task}: {val}')
        assignment_cost[worker, task] = cost
if len(assignment_cost) != num_workers * num_tasks:
    raise ValueError('Assignment cost matrix is incomplete.')
m = gp.Model('worker_task_assignment')
m.setParam('MIPGap', 0.0001)
x_vars = m.addVars([(w, t) for w in worker_ids for t in task_ids], vtype=GRB.BINARY, name='')
y_vars = m.addVars(worker_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((assignment_cost[w, t] * x_vars[w, t] for w in worker_ids for t in task_ids)), GRB.MINIMIZE)
for t in task_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in worker_ids)) == 1, name='task_assign_' + t)
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in task_ids)) <= y_vars[w], name='worker_assign_' + w)
m.addConstr(gp.quicksum((y_vars[w] for w in worker_ids)) == 10, name='select_10_workers')
for w in worker_ids:
    for t in task_ids:
        m.addConstr(x_vars[w, t] <= y_vars[w], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')