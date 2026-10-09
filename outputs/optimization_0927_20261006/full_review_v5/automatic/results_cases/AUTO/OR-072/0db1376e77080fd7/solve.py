import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Shift' not in df.columns or 'Number Required' not in df.columns:
    raise KeyError("Required columns 'Shift' and 'Number Required' not found in CSV.")
df['Shift'] = df['Shift'].str.strip().astype(int)
df['Number Required'] = df['Number Required'].str.strip().astype(int)
shifts = sorted(df['Shift'].unique())
if len(shifts) != 24 or min(shifts) != 1 or max(shifts) != 24:
    raise ValueError('Expected 24 unique shifts numbered 1 to 24.')
required_dict = dict(zip(df['Shift'], df['Number Required']))
m = gp.Model('BusCrewScheduling')
x_vars = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in shifts)), gp.GRB.MINIMIZE)
for s in shifts:
    covering_starts = []
    for t in shifts:
        covered = [(t + offset - 1) % 24 + 1 for offset in range(4)]
        if s in covered:
            covering_starts.append(t)
    if not covering_starts:
        raise ValueError(f'No shift start covers shift {s}.')
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_starts)) >= required_dict[s], name=f'cover_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum total drivers/crew assigned)')
    print('--- Shift Start Assignments ---')
    for t in shifts:
        val = x_vars[t].X
        if val > 1e-06:
            print(f'  Start at hour {t}: {int(round(val))} drivers/crew')
else:
    print(f'No optimal solution found. Status: {m.status}')