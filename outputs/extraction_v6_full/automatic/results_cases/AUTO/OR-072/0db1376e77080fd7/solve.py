import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if set(df['Shift']) != set(range(1, 25)):
    raise ValueError('Shift column must contain exactly the integers 1 to 24.')
required = df.set_index('Shift')['Number Required'].to_dict()
shifts = list(range(1, 25))
m = gp.Model('BusRouteStaffing')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='x')
m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
for h in shifts:
    covering_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= required[h], name=f'cover_{h}')
m.optimize()