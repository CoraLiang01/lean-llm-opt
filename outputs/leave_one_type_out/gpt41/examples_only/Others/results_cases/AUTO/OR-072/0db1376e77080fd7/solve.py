import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if df.shape[0] != 24:
    raise ValueError(f'Expected 24 rows for 24 hours, got {df.shape[0]} rows.')
shifts = df['Shift'].astype(int).tolist()
if sorted(shifts) != list(range(1, 25)):
    raise ValueError('Shift column must enumerate 1..24 exactly once each.')
required_staff = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
time_labels = dict(zip(df['Shift'].astype(int), df['Time'].astype(str)))
m = gp.Model('BusStaffScheduling')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for h in shifts:
    covered_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[tau] for tau in covered_starts)) >= required_staff[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total number of staff assigned: {int(round(m.objVal))}')
    print('\n--- Shift Assignment ---')
    for t in shifts:
        val = int(round(x[t].X))
        if val > 0:
            print(f'Start at {time_labels[t]} (Shift {t}): {val} staff assigned')
else:
    print(f'No optimal solution found. Status: {m.status}')