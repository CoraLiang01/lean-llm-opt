import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
if df.shape[0] != 1:
    raise ValueError(f'Expected exactly one row in 18.csv, got {df.shape[0]}')
row = df.iloc[0]
try:
    bi_dict = {tid: float(row[tid]) for tid in task_ids}
except KeyError as e:
    raise KeyError(f'Missing task column in CSV: {e}')
except ValueError as e:
    raise ValueError(f'Non-numeric BI value in CSV: {e}')
cpu_ids = [1, 2, 3]
cpu_freqs = {1: 1.33, 2: 2.0, 3: 2.66}
m = gp.Model('TaskAssignmentMinMakespan')
x_vars = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
C_p_vars = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.setObjective(C_max_var, gp.GRB.MINIMIZE)
for t in task_ids:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C_p_vars[p] == gp.quicksum((bi_dict[t] / cpu_freqs[p] * x_vars[t, p] for t in task_ids)), name=f'cpu_time_{p}')
for p in cpu_ids:
    m.addConstr(C_p_vars[p] <= C_max_var, name=f'makespan_{p}')
m.optimize()