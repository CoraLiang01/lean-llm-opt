import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if df['Shift'].nunique() != 24 or df.shape[0] != 24:
    raise ValueError('Expected 24 shifts (hours), got {} unique and {} rows.'.format(df['Shift'].nunique(), df.shape[0]))
shifts = sorted(df['Shift'].astype(int).tolist())
required = df.set_index(df['Shift'].astype(int))['Number Required'].to_dict()
m = gp.Model('BusCrewScheduling')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for h in shifts:
    working_starts = [(h - i - 1) % 24 + 1 for i in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in working_starts)) >= required[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum staff assigned)')
    print('--- Staff Assignment Plan ---')
    for t in shifts:
        val = x[t].X
        if val > 1e-06:
            print(f'  Start at hour {t}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')