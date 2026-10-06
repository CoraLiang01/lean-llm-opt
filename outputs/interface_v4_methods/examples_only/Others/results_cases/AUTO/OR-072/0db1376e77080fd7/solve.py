import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
df['Shift'] = df['Shift'].astype(int)
df = df.sort_values('Shift').reset_index(drop=True)
shifts = df['Shift'].tolist()
n_periods = len(shifts)
if n_periods != 24:
    raise ValueError('Expected 24 time periods (hours), got {}'.format(n_periods))
required = dict(zip(df['Shift'], df['Number Required']))
m = gp.Model('BusRouteStaffing')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for h in shifts:
    covering_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= required[h], name=f'cover_{h}')
m.optimize()