import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx = {t: i for i, t in enumerate(periods)}
requirements = dict(zip(df['Time'], df['Requirement']))
shift_len = 16
num_periods = 48
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    t_idx = period_idx[t]
    covered_starts = []
    for s_idx in range(num_periods):
        if (t_idx - s_idx) % num_periods < shift_len:
            covered_starts.append(periods[s_idx])
    m.addConstr(gp.quicksum((x[s] for s in covered_starts)) >= requirements[t], name=f'cover_{t_idx}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for t in periods:
        val = x[t].X
        if val > 0.5:
            print(f"Shift start at '{t}': {int(round(val))} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')