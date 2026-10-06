import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not set(['Shift', 'Number Required']).issubset(df.columns):
    raise KeyError("CSV missing required columns 'Shift' and/or 'Number Required'.")
shifts = sorted(df['Shift'].astype(int).unique())
if shifts != list(range(1, 25)):
    raise ValueError(f'Expected shifts 1..24, got {shifts}')
required = df.set_index(df['Shift'].astype(int))['Number Required'].to_dict()
if len(required) != 24:
    raise ValueError('Missing required staff data for some shifts.')
m = gp.Model('BusStaffScheduling')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[i] for i in shifts)), gp.GRB.MINIMIZE)
for j in shifts:
    covered_starts = [(j - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[i] for i in covered_starts)) >= required[j], name=f'cov{j}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in shifts:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')