import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not set(['Shift', 'Number Required']).issubset(df.columns):
    raise KeyError("CSV missing required columns 'Shift' and/or 'Number Required'.")
shifts = sorted(df['Shift'].astype(int).unique())
if len(shifts) != 24 or min(shifts) != 1 or max(shifts) != 24:
    raise ValueError('Expected 24 shifts indexed from 1 to 24.')
required = df.set_index(df['Shift'].astype(int))['Number Required'].to_dict()
for t in range(1, 25):
    if t not in required:
        raise KeyError(f'Missing required staff for shift {t}.')
m = gp.Model('BusShiftCover')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for h in shifts:
    covering_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= required[h], name=f'cov{h}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for t in shifts:
        print(f'{x[t].VarName} {x[t].X}')
else:
    print(f'Solver status: {m.status}')