import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 task columns (A-J), found: {task_cols}')
tasks = [col.strip() for col in task_cols]
worker_col = 'Task Time Required'
if worker_col in df.columns:
    df[worker_col] = df[worker_col].astype(str).str.strip()
    worker_rows = []
    worker_ids = []
    for (idx, val) in enumerate(df[worker_col]):
        m = re.match('^(?:Worker\\s*)?(\\d{1,2})$', val, re.IGNORECASE)
        if m:
            worker_num = int(m.group(1))
            worker_rows.append(idx)
            worker_ids.append(f'Worker {worker_num}')
    if len(worker_rows) != 12:
        raise ValueError(f'Expected 12 workers, found {len(worker_rows)}: {worker_ids}')
    df_workers = df.iloc[worker_rows].copy()
    df_workers.index = worker_ids
else:
    if df.shape[0] != 12:
        raise ValueError(f'Expected 12 worker rows, found {df.shape[0]}')
    worker_ids = [f'Worker {i + 1}' for i in range(df.shape[0])]
    df_workers = df.copy()
    df_workers.index = worker_ids
workers = worker_ids
cost = {}
for w in workers:
    for t in tasks:
        val = df_workers.loc[w, t]
        if pd.isnull(val):
            raise ValueError(f'Missing time for worker {w}, task {t}')
        cost[w, t] = float(val)

def solve_assignment_problem(workers, tasks, cost):
    m = gp.Model('WorkerTaskAssignment')
    x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    for t in tasks:
        m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'task_{t}')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= y[w], name=f'worker_assign_{w}')
    m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
    for w in workers:
        for t in tasks:
            m.addConstr(x[w, t] <= y[w], name=f'link_{w}_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_assignment_problem(workers, tasks, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')