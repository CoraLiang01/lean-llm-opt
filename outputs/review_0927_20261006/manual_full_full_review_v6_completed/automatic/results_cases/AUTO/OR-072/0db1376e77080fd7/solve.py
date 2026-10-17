import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Shift', 'Number Required']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Shift'] = df['Shift'].str.strip().astype(int)
df['Number Required'] = df['Number Required'].str.strip().astype(int)
shifts = sorted(df['Shift'].unique())
if shifts != list(range(1, 25)):
    raise ValueError('Shifts in CSV do not cover all 24 hours as required.')
number_required = df.set_index('Shift')['Number Required'].to_dict()
m = gp.Model('BusCrewScheduling')
x_vars = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in shifts)), gp.GRB.MINIMIZE)
for s in shifts:
    covered_starts = [(s - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covered_starts)) >= number_required[s], name=f'cover_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum total drivers/crew assigned)')
    print('--- Assignment Plan (number starting at each hour) ---')
    for t in shifts:
        if x_vars[t].X > 1e-06:
            print(f'  Hour {t}: {int(round(x_vars[t].X))} assigned')
else:
    print(f'No optimal solution found. Status: {m.status}')