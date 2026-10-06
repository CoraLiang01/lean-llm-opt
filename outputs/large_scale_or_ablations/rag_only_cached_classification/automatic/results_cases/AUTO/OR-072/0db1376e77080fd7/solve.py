import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not set(['Shift', 'Time', 'Number Required']).issubset(df.columns):
    raise ValueError('CSV missing required columns.')
periods = df['Shift'].astype(int).tolist()
n_periods = len(periods)
if n_periods != 24:
    raise ValueError('Expected 24 periods (hours), got %d' % n_periods)
required = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
m = Model('BusCrewScheduling')
m.Params.OutputFlag = 0
x = m.addVars(periods, vtype=GRB.INTEGER, lb=0, name='')
for t in periods:
    covering_starts = [(t - k - 1) % n_periods + 1 for k in range(4)]
    m.addConstr(sum((x[s] for s in covering_starts)) >= required[t], name=f'cover_{t}')
m.setObjective(x.sum(), GRB.MINIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    total_staff = int(round(m.objVal))
    print(f'Minimum number of drivers and crew members required: {total_staff}')
    print('Assignment (number starting at each period):')
    for t in periods:
        val = int(round(x[t].X))
        print(f"  Period {t} ({df.loc[df['Shift'] == t, 'Time'].values[0]}): {val}")
else:
    print('No optimal solution found.')