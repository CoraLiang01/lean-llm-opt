import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(range(48))
period_idx_to_label = {i: df.iloc[i]['Time'] for i in periods}
try:
    requirement_col = [col for col in df.columns if col.strip().casefold() == 'requirement'][0]
except IndexError:
    raise KeyError("Could not find 'Requirement' column in CSV.")
requirement = {}
for i in periods:
    val = df.iloc[i][requirement_col]
    try:
        requirement[i] = int(val)
    except Exception:
        raise ValueError(f'Invalid requirement value at period {i}: {val}')
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
shift_length = 16
for t in periods:
    covering_starts = [(t - k) % 48 for k in range(shift_length)]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x_vars[s].X for s in periods))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in periods:
        n = int(round(x_vars[s].X))
        if n > 0:
            print(f'  Start at period {s:2d} ({period_idx_to_label[s]}): {n} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')