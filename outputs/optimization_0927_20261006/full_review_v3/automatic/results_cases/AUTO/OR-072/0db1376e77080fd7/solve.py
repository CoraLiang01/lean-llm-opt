import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['Shift'] = df['Shift'].astype(int)
df['Number Required'] = df['Number Required'].astype(int)
df = df.sort_values('Shift')
shifts = df['Shift'].tolist()
n_periods = len(shifts)
if n_periods != 24:
    raise ValueError(f'Expected 24 shifts, got {n_periods}')
required_staff = dict(zip(df['Shift'], df['Number Required']))
m = gp.Model('BusRouteStaffing')
x_vars = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in shifts)), gp.GRB.MINIMIZE)
for s in shifts:
    covered_starts = [(s - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covered_starts)) >= required_staff[s], name=f'cover_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum total staff assigned)')
    print('--- Staff Start Schedule ---')
    for t in shifts:
        val = x_vars[t].X
        if val > 1e-06:
            print(f"  Shift {t} ({df.loc[df['Shift'] == t, 'Time'].values[0]}): {int(round(val))} staff start")
else:
    print(f'No optimal solution found. Status: {m.status}')