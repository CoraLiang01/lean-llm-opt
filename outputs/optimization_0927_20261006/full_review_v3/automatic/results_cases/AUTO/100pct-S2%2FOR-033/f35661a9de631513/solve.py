import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(range(len(df)))
num_periods = len(periods)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
shift_length = 16
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [s for s in periods if (t - s) % num_periods < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments (period index : number of staff) ---')
    for s in periods:
        val = x_vars[s].X
        if val > 1e-06:
            time_label = df.iloc[s]['Time']
            print(f'  Period {s:2d} ({time_label}): {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')