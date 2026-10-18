import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
missing_tasks = [tid for tid in task_ids if tid not in df.columns]
if missing_tasks:
    raise ValueError(f'Missing task columns in CSV: {missing_tasks}')
row_bi = df[df['Process'].str.strip().str.casefold() == 'bi']
if row_bi.shape[0] != 1:
    raise ValueError("Could not find exactly one 'BI' row in the CSV.")
row_bi = row_bi.iloc[0]
bi_dict = {}
for tid in task_ids:
    val = row_bi[tid]
    try:
        bi_dict[tid] = float(val)
    except Exception:
        raise ValueError(f'Invalid BI value for task {tid}: {val}')
proc_time = {}
for tid in task_ids:
    for pid in cpu_ids:
        proc_time[tid, pid] = bi_dict[tid] / cpu_speeds[pid]
m = Model()
assign_vars = m.addVars(task_ids, cpu_ids, vtype=GRB.BINARY, name='')
C_max = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='C_max')
for tid in task_ids:
    m.addConstr(quicksum((assign_vars[tid, pid] for pid in cpu_ids)) == 1)
for pid in cpu_ids:
    m.addConstr(quicksum((proc_time[tid, pid] * assign_vars[tid, pid] for tid in task_ids)) <= C_max)
m.setObjective(C_max, GRB.MINIMIZE)
m.optimize()