import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(path, sep=',')
r = dict(zip(df['Shift'], df['Number Required']))
shifts = sorted(r.keys())
m = gp.Model('BusRouteStaffing')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((x[s] for s in shifts)), gp.GRB.MINIMIZE)
for t in shifts:
    s_start = max(1, t - 3)
    s_end = t
    coverage = gp.quicksum((x[s] for s in range(s_start, s_end + 1)))
    m.addConstr(coverage >= r[t])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total number of workers: {m.objVal:.0f}')
    print('\nStart times and assigned workers:')
    for s in shifts:
        val = x[s].X
        if val > 1e-06:
            time_str = df.loc[df['Shift'] == s, 'Time'].values[0]
            print(f'  Shift {s} (start at {time_str}): {int(val)} workers')
else:
    print(f'Optimization ended with status {m.status}')