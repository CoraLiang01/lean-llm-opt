import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirements = df['Requirement'].astype(int).to_dict()
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in periods:
    covering_starts = [(t - i) % n_periods for i in range(shift_length)]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.setObjective(x.sum(), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Assignments (period index: number of staff starting) ---')
    for s in periods:
        if x[s].X > 1e-06:
            time_label = df.loc[s, 'Time']
            print(f'Period {s:2d} ({time_label}): {int(round(x[s].X))} staff start')
else:
    print(f'No optimal solution found. Status: {m.status}')