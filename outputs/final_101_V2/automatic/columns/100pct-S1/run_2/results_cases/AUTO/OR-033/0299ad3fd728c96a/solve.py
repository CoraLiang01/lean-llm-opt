import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirement = df['Requirement'].astype(int).to_dict()
shift_starts = periods
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_shifts = []
    for s in shift_starts:
        covered = [(s + i) % n_periods for i in range(shift_length)]
        if t in covered:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments (period index : number of staff) ---')
    for s in shift_starts:
        if x[s].X > 0.5:
            time_label = df.loc[s, 'Time']
            print(f'  Start at period {s:2d} ({time_label}): {int(round(x[s].X))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')