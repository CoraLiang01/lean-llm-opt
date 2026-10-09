import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Shift', 'Number Required'}.issubset(df.columns):
    raise KeyError("CSV must contain 'Shift' and 'Number Required' columns.")
shifts = df['Shift'].astype(int).tolist()
if sorted(shifts) != list(range(1, 25)):
    raise ValueError('Expected 24 shifts indexed 1..24, got: %s' % shifts)
required = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
shift_indices = list(range(1, 25))
m = gp.Model('BusCrewScheduling')
x = m.addVars(shift_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shift_indices)), gp.GRB.MINIMIZE)
for s in shift_indices:
    covered_starts = [(s - i - 1) % 24 + 1 for i in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covered_starts)) >= required[s], name=f'cov{s}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for t in shift_indices:
        print(f'{x[t].VarName} {x[t].X}')
else:
    print(f'Solver status: {m.status}')