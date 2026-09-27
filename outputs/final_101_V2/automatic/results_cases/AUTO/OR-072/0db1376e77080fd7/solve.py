import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not set(['Shift', 'Number Required']).issubset(df.columns):
    raise KeyError("CSV missing required columns 'Shift' and/or 'Number Required'.")
df['Shift'] = df['Shift'].astype(int)
df = df.sort_values('Shift')
shifts = df['Shift'].tolist()
n_periods = len(shifts)
if n_periods != 24:
    raise ValueError('Expected 24 shifts (hours), got %d.' % n_periods)
required = dict(zip(df['Shift'], df['Number Required']))
m = gp.Model('BusCrewScheduling')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for s in shifts:
    covered_by = []
    for dt in range(4):
        t = (s - dt - 1) % 24 + 1
        covered_by.append(t)
    m.addConstr(gp.quicksum((x[t] for t in covered_by)) >= required[s], name=f'cover_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum total drivers/crew assigned)')
    print('--- Assignment Plan ---')
    for t in shifts:
        if x[t].X > 1e-06:
            print(f'  Start at hour {t}: {int(round(x[t].X))} assigned')
else:
    print(f'No optimal solution found. Status: {m.status}')