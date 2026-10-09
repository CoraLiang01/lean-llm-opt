import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
shift_length = 16
for t in periods:
    covering_s = [s for s in periods if (t - s) % n_periods < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_s)) >= requirements[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((int(round(x[s].X)) for s in periods))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Assignments (period index: number of staff starting) ---')
    for s in periods:
        val = int(round(x[s].X))
        if val > 0:
            time_label = df.loc[s, 'Time']
            print(f'  Period {s:2d} ({time_label}): {val} staff start')
else:
    print(f'No optimal solution found. Status: {m.status}')