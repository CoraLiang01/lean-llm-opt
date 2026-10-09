import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df.index)
period_idx_to_time = df['Time'].to_dict()
requirement = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
num_periods = len(periods)
shift_length = 16
for t in periods:
    covering_starts = []
    for s in periods:
        covered = [(int(s) + offset) % num_periods for offset in range(shift_length)]
        if int(t) in covered:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments ---')
    for s in periods:
        val = x_vars[s].X
        if val > 1e-06:
            print(f'Start at period {s} ({period_idx_to_time[s]}): {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')