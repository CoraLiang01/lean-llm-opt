import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = {'Shift', 'Time', 'Number Required'}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(f'Missing required columns in CSV: {missing}')
shifts = df['Shift'].astype(int).tolist()
if sorted(shifts) != list(range(1, 25)):
    raise ValueError('Shift indices must be consecutive integers from 1 to 24.')
required_staff = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
m = gp.Model('BusRouteStaffing')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for h in shifts:
    covered_starts = [(h - i - 1) % 24 + 1 for i in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covered_starts)) >= required_staff[h], name=f'cov{h}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for t in shifts:
        print(f'{x[t].VarName} {x[t].X}')
else:
    print(f'Solver status: {m.status}')