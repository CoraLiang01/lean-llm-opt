import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise KeyError("CSV must contain 'Time' and 'Requirement' columns.")
period_labels = df['Time'].astype(str).tolist()
requirements = df['Requirement'].astype(int).tolist()
n_periods = len(period_labels)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods (half-hours in 24h), got {n_periods}.')
periods = list(range(n_periods))
shifts = list(range(n_periods))
shift_length = 16
cover = np.zeros((n_periods, n_periods), dtype=int)
for t in periods:
    for s in shifts:
        if (t - s) % n_periods < shift_length:
            cover[t, s] = 1
m = gp.Model('MinWaitstaffShifts')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shifts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in shifts if cover[t, s])) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((x[s].X for s in shifts))
    print(f'Optimal total value/cost: {total_waitstaff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shifts:
        num = int(round(x[s].X))
        if num > 0:
            print(f"Start at '{period_labels[s]}': {num} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')