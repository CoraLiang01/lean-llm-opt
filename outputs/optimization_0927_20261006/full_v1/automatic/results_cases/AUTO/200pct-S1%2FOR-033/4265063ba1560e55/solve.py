import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirement = {}
for (idx, row) in df.iterrows():
    try:
        req = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement at row {idx}: {row['Requirement']}") from e
    requirement[idx] = req
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [s for s in periods if (t - s) % n_periods in range(16)]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Assignments (period index: number of staff starting) ---')
    for s in periods:
        val = x_vars[s].X
        if val > 1e-06:
            time_label = df.loc[s, 'Time']
            print(f'  Period {s:2d} ({time_label}): {int(round(val))} staff start')
else:
    print(f'No optimal solution found. Status: {m.status}')