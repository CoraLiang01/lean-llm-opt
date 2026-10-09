import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
missing_tasks = [tid for tid in task_ids if tid not in df.columns]
if missing_tasks:
    raise ValueError(f'Missing task columns in CSV: {missing_tasks}')
bi_row = df[df['Process'].str.casefold().str.strip() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one 'BI' row in the CSV.")
task_bi = {}
for tid in task_ids:
    val = bi_row.iloc[0][tid]
    try:
        task_bi[tid] = float(val)
    except Exception:
        raise ValueError(f'Non-numeric or missing BI value for task {tid}: {val}')
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
m = Model('makespan_parallel_machine')
assign_vars = m.addVars(task_ids, cpu_ids, vtype=GRB.BINARY, name='')
C_max = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='C_max')
for t in task_ids:
    m.addConstr(quicksum((assign_vars[t, p] for p in cpu_ids)) == 1, name='')
for p in cpu_ids:
    m.addConstr(quicksum((task_bi[t] / cpu_speeds[p] * assign_vars[t, p] for t in task_ids)) <= C_max, name='')
m.setObjective(C_max, GRB.MINIMIZE)
m.optimize()