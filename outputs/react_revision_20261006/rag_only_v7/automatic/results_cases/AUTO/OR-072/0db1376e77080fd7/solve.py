import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if set(['Shift', 'Time', 'Number Required']) - set(df.columns):
    raise ValueError('Missing required columns in CSV.')
df['Shift'] = df['Shift'].astype(int)
df['Number Required'] = df['Number Required'].astype(int)
periods = list(df['Shift'])
if sorted(periods) != list(range(1, 25)):
    raise ValueError('Expected 24 consecutive periods indexed 1..24.')
required_staff = dict(zip(df['Shift'], df['Number Required']))
shift_starts = periods
coverage = {s: [] for s in periods}
for t in shift_starts:
    covered = [(t + offset - 1) % 24 + 1 for offset in range(4)]
    for s in covered:
        coverage[s].append(t)
for s in periods:
    if not coverage[s]:
        raise ValueError(f'Period {s} is not covered by any shift start.')
m = gp.Model('bus_staff_scheduling')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(shift_starts, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in shift_starts)), GRB.MINIMIZE)
for s in periods:
    m.addConstr(gp.quicksum((x_vars[t] for t in coverage[s])) >= required_staff[s], name=f'cover_{s}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for t in shift_starts:
        print(f'x[{t}]: {x_vars[t].X}')
else:
    print(f'Solver status: {m.Status}')