import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
coverage = {t: [s for s in periods if (t - s) % n_periods in range(16)] for t in periods}
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in coverage[t])) >= requirements[t], name=f'cover_{t}')
m.setObjective(x.sum(), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in periods))
    print(f'Optimal total value/cost: {total_staff:.0f} (minimum number of waitstaff)')
    print('\n--- Shift Start Assignments (period index : staff count) ---')
    for s in periods:
        if x[s].X > 1e-06:
            time_label = df.loc[s, 'Time']
            print(f'  Start at period {s:2d} ({time_label}): {int(round(x[s].X))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')