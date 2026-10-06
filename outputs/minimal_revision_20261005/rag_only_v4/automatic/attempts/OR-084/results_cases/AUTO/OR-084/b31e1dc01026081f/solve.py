import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_ids = [str(i) for i in range(1, 41)]
missing_cols = [tid for tid in task_ids if tid not in df.columns]
if missing_cols:
    raise ValueError(f'Missing task columns in CSV: {missing_cols}')
df['Process_norm'] = df['Process'].astype(str).str.casefold().str.strip()
bi_row = df[df['Process_norm'] == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Expected exactly one row with Process == 'BI' (case-insensitive, trimmed)")
bi_row = bi_row.iloc[0]
BI = {}
for tid in task_ids:
    val = bi_row[tid]
    if pd.isnull(val):
        raise ValueError(f'Missing BI value for task {tid}')
    BI[tid] = float(val)
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = BI[t] / cpu_speeds[p]
for (k, v) in proc_time.items():
    if v <= 0:
        raise ValueError(f'Non-positive processing time for {k}: {v}')
m = gp.Model('makespan_min')
assign = m.addVars([(t, p) for t in task_ids for p in cpu_ids], vtype=GRB.BINARY, name='')
Cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in task_ids:
    m.addConstr(gp.quicksum((assign[t, p] for p in cpu_ids)) == 1, name='')
for p in cpu_ids:
    m.addConstr(gp.quicksum((assign[t, p] * proc_time[t, p] for t in task_ids)) <= Cmax, name='')
m.setObjective(Cmax, GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')