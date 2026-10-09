import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and 'Requirement' not found in CSV.")
period_indices = list(range(len(df)))
if len(period_indices) != 48:
    raise ValueError(f'Expected 48 periods, found {len(period_indices)}.')
period_labels = df['Time'].tolist()
try:
    requirement_series = df['Requirement'].astype(int)
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
requirement = dict(zip(period_indices, requirement_series))
shift_length = 16
num_periods = 48
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(period_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in period_indices)), gp.GRB.MINIMIZE)
for t in period_indices:
    covering_starts = []
    for s in period_indices:
        if (t - s) % num_periods < shift_length:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift starts cover period {t} ({period_labels[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in period_indices:
        val = x_vars[s].X
        if val > 1e-06:
            print(f'  Start at {period_labels[s]}: {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')