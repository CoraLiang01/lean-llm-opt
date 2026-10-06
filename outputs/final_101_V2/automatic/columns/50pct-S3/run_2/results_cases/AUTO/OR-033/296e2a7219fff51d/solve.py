import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
requirements = df['Requirement'].astype(int).to_dict()
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [s for s in periods if (t - s) % 48 in range(shift_length)]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in periods))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in periods:
        n_staff = int(round(x[s].X))
        if n_staff > 0:
            time_label = df.iloc[s]['Time']
            print(f"Start at '{time_label}': {n_staff} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')