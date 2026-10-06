import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if set(df['Shift']) != set(range(1, 25)):
    raise ValueError('Shift column must contain all integers from 1 to 24 inclusive.')
shifts = sorted(df['Shift'].astype(int).tolist())
n_hours = 24
required = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
m = gp.Model('BusRouteStaffScheduling')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for h in shifts:
    covering_starts = [(h - k - 1) % n_hours + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= required[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum total staff assigned)')
    print('\n--- Shift Start Assignments (x[t]) ---')
    for t in shifts:
        if x[t].X > 1e-06:
            print(f"  Hour {t:2d} ({df.loc[df['Shift'] == t, 'Time'].values[0]}): {int(round(x[t].X))} staff assigned")
else:
    print(f'No optimal solution found. Status: {m.status}')