import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [str(i) for i in range(1, 41)]
if not (df['Process'] == 'BI').any():
    raise ValueError("No row with Process == 'BI' found in 18.csv")
bi_row = df[df['Process'] == 'BI'].iloc[0]
task_bi = {}
for t in task_cols:
    val = bi_row[t]
    try:
        task_bi[t] = float(val)
    except Exception:
        raise ValueError(f'Invalid BI value for task {t}: {val}')
cpu_ids = ['1', '2', '3']
cpu_freq = {'1': 1.33, '2': 2.0, '3': 2.66}
m = gp.Model('ParallelMachineMakespan')
x_vars = m.addVars(task_cols, cpu_ids, vtype=gp.GRB.BINARY, name='')
C_vars = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='C_max')
for t in task_cols:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C_vars[p] == gp.quicksum((task_bi[t] / cpu_freq[p] * x_vars[t, p] for t in task_cols)), name=f'cpu_time_{p}')
for p in cpu_ids:
    m.addConstr(C_vars[p] <= C_max_var, name=f'makespan_{p}')
m.setObjective(C_max_var, gp.GRB.MINIMIZE)
m.optimize()