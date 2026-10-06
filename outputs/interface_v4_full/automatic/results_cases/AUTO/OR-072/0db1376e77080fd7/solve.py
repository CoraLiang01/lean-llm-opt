import gurobipy as gp
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv', sep=',')
periods = df['Shift'].astype(int).tolist()
req = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
if set(periods) != set(range(1, 25)):
    raise ValueError('Shift periods in CSV do not cover all 24 hours (1..24).')
m = gp.Model('BusRouteShiftScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
for s in periods:
    covering_starts = [(s - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= req[s], name=f'cover_{s}')
m.optimize()