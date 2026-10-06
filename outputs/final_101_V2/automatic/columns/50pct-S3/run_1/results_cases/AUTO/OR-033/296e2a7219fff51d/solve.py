import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)} from CSV.')
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
shift_length = 16
for t in periods:
    covering_starts = [(t - i) % 48 for i in range(shift_length)]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in periods:
        if x[s].X > 1e-06:
            time_label = df.iloc[s]['Time']
            print(f'Shift start at period {s} ({time_label}): {int(round(x[s].X))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')