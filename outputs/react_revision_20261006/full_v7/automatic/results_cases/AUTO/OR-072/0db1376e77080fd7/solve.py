import gurobipy as gp
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv', dtype=str, keep_default_na=False)
if df.shape[0] != 24:
    raise ValueError(f'Expected 24 rows for 24 periods, got {df.shape[0]}.')
try:
    df['Shift'] = df['Shift'].astype(int)
    df['Number Required'] = df['Number Required'].astype(int)
except Exception as e:
    raise ValueError(f'Failed to convert Shift or Number Required to int: {e}')
shifts = sorted(df['Shift'].unique())
if shifts != list(range(1, 25)):
    raise ValueError(f'Shift indices must be 1..24, got {shifts}')
required = dict(zip(df['Shift'], df['Number Required']))
m = gp.Model('BusShiftCover')
x_vars = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in shifts)), gp.GRB.MINIMIZE)
for s in shifts:
    covering_ts = [t for t in shifts if (s - t) % 24 in [0, 1, 2, 3]]
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_ts)) >= required[s], name=f'cover_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for t in shifts:
        print(f'{x_vars[t].VarName} {x_vars[t].X}')
else:
    print(f'Solver status: {m.status}')