import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirement = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
shift_length = 16
for p in periods:
    covering_starts = [t for t in periods if (p - t) % n_periods < shift_length]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= requirement[p], name=f'cover_{p}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[t].X for t in periods))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for t in periods:
        n = int(round(x[t].X))
        if n > 0:
            time_label = df.loc[t, 'Time']
            print(f'  Start at period {t:2d} ({time_label}): {n} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')