import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not set(['Shift', 'Time', 'Number Required']).issubset(df.columns):
    raise ValueError('CSV missing required columns.')
shifts = df['Shift'].astype(int).tolist()
if sorted(shifts) != list(range(1, 25)):
    raise ValueError('Shift column must contain all integers from 1 to 24, one per hour.')
required_staff = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
T = 24
shift_indices = list(range(1, T + 1))
m = gp.Model('BusRouteStaffing')
x = m.addVars(shift_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shift_indices)), gp.GRB.MINIMIZE)
for h in shift_indices:
    covered_starts = [(h - k - 1) % T + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covered_starts)) >= required_staff[h], name=f'cover_{h}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for t in shift_indices:
        print(f'{x[t].VarName} {x[t].X}')
else:
    print(f'Solver status: {m.status}')