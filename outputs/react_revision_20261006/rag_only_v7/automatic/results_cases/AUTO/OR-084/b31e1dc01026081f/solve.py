import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
missing_tasks = [tid for tid in task_ids if tid not in df.columns]
if missing_tasks:
    raise ValueError(f'Missing task columns in CSV: {missing_tasks}')
if not (df.shape[0] == 1 and df['Process'].str.casefold().str.strip().iloc[0] == 'bi'):
    raise ValueError("CSV must contain exactly one row with Process == 'BI' (case-insensitive, trimmed)")
bi_row = df.iloc[0]
bi_dict = {}
for tid in task_ids:
    try:
        bi_dict[tid] = float(bi_row[tid])
    except Exception as e:
        raise ValueError(f'Invalid BI value for task {tid}: {bi_row[tid]}') from e
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = bi_dict[t] / cpu_speeds[p]
m = gp.Model('makespan_sched')
m.Params.MIPGap = 0.0001
assign_vars = m.addVars([(t, p) for t in task_ids for p in cpu_ids], vtype=GRB.BINARY, name='')
cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='cmax')
for t in task_ids:
    m.addConstr(gp.quicksum((assign_vars[t, p] for p in cpu_ids)) == 1, name='')
for p in cpu_ids:
    m.addConstr(gp.quicksum((proc_time[t, p] * assign_vars[t, p] for t in task_ids)) <= cmax, name='')
m.setObjective(cmax, GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')