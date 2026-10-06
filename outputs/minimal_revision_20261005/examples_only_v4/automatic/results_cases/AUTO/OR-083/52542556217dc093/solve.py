import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError(f'Expected 10 task columns (A-J), found: {task_cols}')

def is_worker_id(val):
    try:
        return str(int(str(val).strip())) == str(val).strip()
    except Exception:
        return False
worker_rows = df[df['Task Time Required'].apply(is_worker_id)].copy()
worker_ids = worker_rows['Task Time Required'].astype(str).tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found: {worker_ids}')
cost = {}
for (idx, row) in worker_rows.iterrows():
    w = str(row['Task Time Required'])
    for t in task_cols:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time for worker {w}, task {t}')
        cost[w, t] = float(val)
if len(cost) != 12 * 10:
    raise ValueError('Cost matrix is incomplete.')

def solve_assignment_problem(worker_ids, task_cols, cost):
    m = gp.Model('worker_task_assignment')
    m.Params.MIPGap = 0.0001
    x = m.addVars([(w, t) for w in worker_ids for t in task_cols], vtype=gp.GRB.BINARY, name='')
    y = m.addVars(worker_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in worker_ids for t in task_cols)), gp.GRB.MINIMIZE)
    for t in task_cols:
        m.addConstr(gp.quicksum((x[w, t] for w in worker_ids)) == 1, name=f'task_{t}_assign')
    for w in worker_ids:
        m.addConstr(gp.quicksum((x[w, t] for t in task_cols)) <= y[w], name=f'worker_{w}_assign')
    m.addConstr(gp.quicksum((y[w] for w in worker_ids)) == 10, name='select_10_workers')
    for w in worker_ids:
        m.addConstr(gp.quicksum((x[w, t] for t in task_cols)) <= 1, name=f'worker_{w}_one_task')
    for w in worker_ids:
        for t in task_cols:
            m.addConstr(x[w, t] <= y[w], name=f'link_{w}_{t}')
    m.optimize()
    return m
m = solve_assignment_problem(worker_ids, task_cols, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')