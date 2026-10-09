import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and 'Requirement' not found in 44.csv")
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, found {len(periods)} in 44.csv')
try:
    requirements = {p: int(df.loc[p, 'Requirement']) for p in periods}
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' to int for all periods: {e}")
shift_length = 16
num_periods = 48
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [s for s in periods if (t - s) % num_periods < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x_vars[s].X)) for s in periods))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Schedule ---')
    for s in periods:
        num = int(round(x_vars[s].X))
        if num > 0:
            time_label = df.loc[s, 'Time']
            print(f'  Start at period {s} ({time_label}): {num} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')