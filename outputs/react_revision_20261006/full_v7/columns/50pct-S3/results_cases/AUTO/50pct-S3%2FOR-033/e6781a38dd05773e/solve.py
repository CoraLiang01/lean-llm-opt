import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 48:
    raise ValueError(f'Expected 48 rows for 48 half-hour periods, got {df.shape[0]}.')
periods = list(df.index)
period_labels = df['Time'].tolist()
try:
    requirements = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError("Could not convert 'Requirement' column to int.") from e
num_periods = 48
shift_length = 16
shift_start_indices = list(range(num_periods))
coverage_s = {t: [] for t in range(num_periods)}
for s in shift_start_indices:
    for offset in range(shift_length):
        t = (s + offset) % num_periods
        coverage_s[t].append(s)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
for t in range(num_periods):
    m.addConstr(gp.quicksum((x_vars[s] for s in coverage_s[t])) >= requirements[t], name=f'cov{t}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_start_indices:
        print(f'{x_vars[s].VarName} {x_vars[s].X}')
else:
    print(f'Solver status: {m.status}')