import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
requirement = df['Requirement'].astype(int).to_dict()
shift_starts = list(range(n_periods))
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_shifts = []
    for s in shift_starts:
        covered_periods = [(s + offset) % n_periods for offset in range(shift_length)]
        if t in covered_periods:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_starts:
        if x[s].X > 0.5:
            time_label = df.loc[s, 'Time']
            print(f'Start at period {s} ({time_label}): {int(round(x[s].X))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')