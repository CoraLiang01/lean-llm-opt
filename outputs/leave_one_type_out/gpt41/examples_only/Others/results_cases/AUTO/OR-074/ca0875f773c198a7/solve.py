import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)} from CSV.')
requirements = df['Requirement'].astype(int).to_dict()
shift_length = 16
m = gp.Model('MinWaitstaff')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [s for s in periods if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x[s].X)) for s in periods))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in periods:
        n = int(round(x[s].X))
        if n > 0:
            time_label = df.loc[s, 'Time']
            print(f'Start at {time_label}: {n} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')