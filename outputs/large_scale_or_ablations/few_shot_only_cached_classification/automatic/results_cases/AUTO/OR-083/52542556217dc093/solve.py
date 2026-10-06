import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 task columns (A-J), found: {task_cols}')
df = df.dropna(how='all')
worker_col = 'Task Time Required'
if worker_col not in df.columns:
    raise ValueError("Expected 'Task Time Required' as first column.")

def clean_worker_id(val):
    if isinstance(val, str):
        m = re.match('\\s*Worker\\s*(\\d+)', val, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m2 = re.match('\\s*(\\d+)', val)
        if m2:
            return int(m2.group(1))
        return val.strip()
    elif pd.isnull(val):
        return None
    else:
        return int(val)
df['worker_id'] = df[worker_col].apply(clean_worker_id)
df = df[~df['worker_id'].isnull()]
workers = list(df['worker_id'])
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
tasks = task_cols
cost = {}
for _, row in df.iterrows():
    w = row['worker_id']
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
    m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'task_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= y[w], name=f'worker_assign_{w}')
m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= 1, name=f'worker_one_task_{w}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    assignment = []
    for t in tasks:
        for w in workers:
            if x[w, t].X > 0.5:
                assignment.append((t, w, cost[w, t]))
    assignment.sort()
    for t, w, time in assignment:
        print(f'Task {t}: Worker {w} (Time: {time:.2f})')
    print('--- Selected Workers ---')
    selected_workers = [w for w in workers if y[w].X > 0.5]
    print('Selected workers:', sorted(selected_workers))
else:
    print(f'No optimal solution found. Status: {m.status}')