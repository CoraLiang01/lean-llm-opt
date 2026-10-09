import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 48:
    raise ValueError(f'Expected 48 periods in 44.csv, got {df.shape[0]}')
periods = list(range(48))
shift_starts = list(range(48))
if 'Requirement' not in df.columns:
    raise KeyError("Column 'Requirement' not found in 44.csv")
try:
    requirements = df['Requirement'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to float: {e}")
requirements = {int(idx): float(req) for (idx, req) in zip(df.index, df['Requirement'])}
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [(t - i) % 48 for i in range(shift_length)]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in shift_starts:
        val = x_vars[s].X
        if val > 1e-06:
            print(f'  Shift start at period {s}: {val:.0f} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')