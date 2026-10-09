import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("CSV must contain 'Time' and 'Requirement' columns.")
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 half-hour periods, got {n_periods}.')
period_labels = df['Time'].tolist()
requirements = df['Requirement'].astype(int).tolist()
shift_length_periods = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(n_periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[j] for j in range(n_periods))), gp.GRB.MINIMIZE)
for i in range(n_periods):
    covering_j = []
    for j in range(n_periods):
        covered = [(j + k) % n_periods for k in range(shift_length_periods)]
        if i in covered:
            covering_j.append(j)
    if not covering_j:
        raise ValueError(f'No shift covers period {i} ({period_labels[i]})')
    m.addConstr(gp.quicksum((x[j] for j in covering_j)) >= requirements[i], name=f'cover_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x[j].X)) for j in range(n_periods)))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Schedule ---')
    for j in range(n_periods):
        num = int(round(x[j].X))
        if num > 0:
            print(f'  Start at {period_labels[j]}: {num} waitstaff')
    print('\n--- Coverage by Period ---')
    for i in range(n_periods):
        coverage = sum((int(round(x[j].X)) for j in range(n_periods) if i in [(j + k) % n_periods for k in range(shift_length_periods)]))
        print(f'  {period_labels[i]}: Required={requirements[i]}, Covered={coverage}')
else:
    print(f'No optimal solution found. Status: {m.status}')