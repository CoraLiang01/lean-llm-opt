import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not set(['Shift', 'Time', 'Number Required']).issubset(df.columns):
    raise ValueError('CSV missing required columns.')
periods = df['Shift'].astype(int).tolist()
if sorted(periods) != list(range(1, 25)):
    raise ValueError('Expected 24 consecutive shifts indexed 1..24.')
required = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
m = Model('bus_shift_scheduling')
m.Params.OutputFlag = 0
x = m.addVars(periods, vtype=GRB.INTEGER, lb=0, name='')
for s in periods:
    covered_t = [(s - i - 1) % 24 + 1 for i in range(4)]
    m.addConstr(sum((x[t] for t in covered_t)) >= required[s], name=f'cover_{s}')
m.setObjective(x.sum(), GRB.MINIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    total_staff = int(round(m.objVal))
    print(f'Minimum number of drivers and crew members assigned: {total_staff}')
    print('Assignment (number starting at each period):')
    for t in periods:
        val = int(round(x[t].X))
        print(f"  Period {t} ({df.loc[df['Shift'] == t, 'Time'].values[0]}): {val}")
else:
    print('No optimal solution found.')