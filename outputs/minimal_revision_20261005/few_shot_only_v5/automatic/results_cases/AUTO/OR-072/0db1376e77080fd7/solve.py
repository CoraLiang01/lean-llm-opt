import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
required_cols = {'Shift', 'Time', 'Number Required'}
if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    raise ValueError(f'Missing required columns in CSV: {missing}')
try:
    df['Hour'] = df['Time'].astype(int)
except Exception:
    raise ValueError("Column 'Time' must contain integer hour indices 1-24.")
hours = sorted(df['Hour'].unique())
if hours != list(range(1, 25)):
    raise ValueError(f'CSV must contain exactly one row for each hour 1-24. Found: {hours}')
required_staff = df.set_index('Hour')['Number Required'].to_dict()
hour_indices = list(range(1, 25))
m = gp.Model('BusRouteStaffing')
x = m.addVars(hour_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in hour_indices)), gp.GRB.MINIMIZE)
for h in hour_indices:
    covering_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= required_staff[h], name=f'cov{h}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for t in hour_indices:
        print(f'{x[t].VarName} {x[t].X}')
else:
    print(f'Solver status: {m.status}')