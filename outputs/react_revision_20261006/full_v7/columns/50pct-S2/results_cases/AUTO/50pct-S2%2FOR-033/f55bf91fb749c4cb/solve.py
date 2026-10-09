import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 48:
    raise ValueError(f'Expected 48 periods, got {df.shape[0]} rows in {csv_path}')
period_keys = list(df.index)
time_labels = df['Time'].tolist()
try:
    requirement_series = df['Requirement'].astype(int)
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
requirement = {k: requirement_series.iloc[k] for k in period_keys}
if any((r < 0 for r in requirement.values())):
    raise ValueError("Negative requirement found in 'Requirement' column.")
num_periods = 48
shift_length = 16
shift_start_keys = period_keys
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in period_keys:
    covering_s = []
    for s in shift_start_keys:
        covered = [(s + offset) % num_periods for offset in range(shift_length)]
        if t in covered:
            covering_s.append(s)
    if not covering_s:
        raise ValueError(f'No shift covers period {t} ({time_labels[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_s)) >= requirement[t], name=f'cov_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_keys)), gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for s in shift_start_keys:
        print(f'{x_vars[s].VarName} {x_vars[s].X}')
else:
    print(f'Solver status: {m.Status}')