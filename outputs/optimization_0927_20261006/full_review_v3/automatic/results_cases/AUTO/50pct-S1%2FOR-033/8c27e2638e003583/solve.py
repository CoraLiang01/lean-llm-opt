import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df.index)
period_labels = df['Time'].tolist()
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
shift_length = 16
num_periods = len(periods)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods (half-hours in 24h), got {num_periods}')
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [(t - i) % num_periods for i in range(shift_length)]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x_vars[s].X)) for s in periods))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Assignments ---')
    for s in periods:
        num = int(round(x_vars[s].X))
        if num > 0:
            print(f"  Shift starting at '{period_labels[s]}' : {num} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')