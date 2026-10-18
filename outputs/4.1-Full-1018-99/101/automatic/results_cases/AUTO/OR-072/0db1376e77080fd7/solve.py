import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Shift' not in df.columns or 'Number Required' not in df.columns:
    raise KeyError("Required columns 'Shift' and/or 'Number Required' not found in CSV.")
df['Shift'] = df['Shift'].str.strip().astype(int)
df['Number Required'] = df['Number Required'].str.strip().astype(int)
df = df.sort_values('Shift')
shifts = df['Shift'].tolist()
if len(shifts) != 24 or set(shifts) != set(range(1, 25)):
    raise ValueError('Expected exactly 24 unique shifts labeled 1..24.')
required_staff = dict(zip(df['Shift'], df['Number Required']))
m = gp.Model('BusRouteStaffing')
x_vars = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in shifts)), gp.GRB.MINIMIZE)
for t in shifts:
    covered_shifts = [(t - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[s] for s in covered_shifts)) >= required_staff[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum total staff assigned)')
    print('--- Staff Assignment Plan ---')
    for t in shifts:
        if x_vars[t].X > 1e-06:
            print(f"  Shift {t} (Start {df.loc[df['Shift'] == t, 'Time'].values[0]}): {int(round(x_vars[t].X))} staff start")
else:
    print(f'No optimal solution found. Status: {m.status}')