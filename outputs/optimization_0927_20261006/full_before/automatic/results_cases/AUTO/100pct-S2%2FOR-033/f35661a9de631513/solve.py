import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(48))
shift_starts = list(range(48))
if len(df) != 48:
    raise ValueError(f'Expected 48 periods, got {len(df)} rows in CSV.')
requirements = df['Requirement'].astype(int).to_dict()
shift_length = 16
m = gp.Model('MinWaitstaff')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = []
    for s in shift_starts:
        covered = [(s + offset) % 48 for offset in range(shift_length)]
        if t in covered:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift starts cover period {t}')
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x[s].X)) for s in shift_starts))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Assignment ---')
    for s in shift_starts:
        num = int(round(x[s].X))
        if num > 0:
            time_label = df.iloc[s]['Time']
            print(f'Shift start at period {s} ({time_label}): {num} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')