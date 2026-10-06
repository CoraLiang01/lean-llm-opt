import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if df.shape[0] != 24:
    raise ValueError(f'Expected 24 rows for 24 hours, got {df.shape[0]} rows.')
required_staff = {}
for idx, row in df.iterrows():
    t = int(row['Shift'])
    required = int(row['Number Required'])
    required_staff[t] = required
m = gp.Model('BusRouteStaffScheduling')
hours = list(range(1, 25))
x = m.addVars(hours, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in hours)), gp.GRB.MINIMIZE)
for h in hours:
    covered_starts = [(h - i - 1) % 24 + 1 for i in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covered_starts)) >= required_staff[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum total staff assigned)')
    print('--- Staff Assignment Plan ---')
    for t in hours:
        val = x[t].X
        if val > 1e-06:
            print(f'  Start at hour {t}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')