import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
shift_length = 16
for t in periods:
    covering_starts = [(t - i) % n_periods for i in range(shift_length)]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((x[t].X for t in periods))
    print(f'Optimal total value/cost: {total_waitstaff:.0f} (minimum number of waitstaff)')
    print('\n--- Shift Start Assignments (period index: number of staff starting) ---')
    for t in periods:
        val = int(round(x[t].X))
        if val > 0:
            time_label = df.loc[t, 'Time']
            print(f'Period {t:2d} ({time_label}): {val} staff start')
else:
    print(f'No optimal solution found. Status: {m.status}')