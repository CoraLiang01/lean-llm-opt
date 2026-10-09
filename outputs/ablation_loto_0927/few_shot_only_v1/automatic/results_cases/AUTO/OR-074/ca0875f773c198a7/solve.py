import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise KeyError("44.csv must contain columns 'Time' and 'Requirement'.")
periods = list(df.index)
if len(periods) != 48:
    raise ValueError('Expected 48 periods (rows) in 44.csv, got {}'.format(len(periods)))
requirements = df['Requirement'].astype(int).to_dict()
shift_starts = periods
shift_length = 16
coverage_s = {t: [] for t in periods}
for s in shift_starts:
    covered = [(s + i) % 48 for i in range(shift_length)]
    for t in covered:
        coverage_s[t].append(s)
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in coverage_s[t])) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_starts:
        val = x[s].X
        if val > 1e-06:
            time_label = df.loc[s, 'Time'] if 'Time' in df.columns else f'Period {s}'
            print(f'  Start at {time_label}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')