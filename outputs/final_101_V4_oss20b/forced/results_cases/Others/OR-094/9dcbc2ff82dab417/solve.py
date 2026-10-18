import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
df = pd.read_csv(path, sep=',')
workstations = df['Workstation'].tolist()
maintenance_percent = dict(zip(workstations, df['Maintenance_Percent']))
effective_capacity = {w: 1440 * (1 - maintenance_percent[w] / 100) for w in workstations}
time_cols = [col for col in df.columns if col.startswith('HiFi') and col.endswith('_Minutes')]
t = {}
for w in workstations:
    row = df.loc[df['Workstation'] == w].iloc[0]
    t[w] = {}
    for col in time_cols:
        i = int(re.search('HiFi(\\d+)_Minutes', col).group(1))
        t[w][i] = float(row[col])
models = sorted(t[workstations[0]].keys())
m = gp.Model('MinimizeIdleTime')
x = m.addVars(models, vtype=gp.GRB.INTEGER)
for w in workstations:
    m.addConstr(gp.quicksum((t[w][i] * x[i] for i in models)) <= effective_capacity[w])
const_term = sum((effective_capacity[w] for w in workstations))
used_time_expr = gp.quicksum((t[w][i] * x[i] for w in workstations for i in models))
m.setObjective(const_term - used_time_expr, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total idle time: {m.objVal:.2f} minutes')
    print('\nProduction plan (units per model):')
    for i in models:
        qty = int(round(x[i].X))
        if qty > 0:
            print(f'  HiFi-{i}: {qty}')
    print('\nIdle time per workstation:')
    for w in workstations:
        used = sum((t[w][i] * x[i].X for i in models))
        idle = effective_capacity[w] - used
        print(f'  Workstation {w}: {idle:.2f} minutes')
else:
    print(f'No optimal solution found. Status: {m.status}')