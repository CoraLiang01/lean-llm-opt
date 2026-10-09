import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and/or 'Requirement' not found in CSV.")
time_periods = df['Time'].tolist()
n_periods = len(time_periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 time periods, got {n_periods}.')
requirement = {}
for (idx, row) in df.iterrows():
    period_id = row['Time']
    try:
        req = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement value at period '{period_id}': {row['Requirement']}") from e
    requirement[period_id] = req
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(time_periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in time_periods)), gp.GRB.MINIMIZE)
period_idx = {t: i for (i, t) in enumerate(time_periods)}
for p in time_periods:
    p_idx = period_idx[p]
    covering_starts = []
    for t in time_periods:
        t_idx = period_idx[t]
        covered_indices = [(t_idx + offset) % n_periods for offset in range(16)]
        if p_idx in covered_indices:
            covering_starts.append(t)
    if not covering_starts:
        raise ValueError(f"No shift covers period '{p}' (index {p_idx})")
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_starts)) >= requirement[p], name=f'cover_{p_idx}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for t in time_periods:
        val = x_vars[t].X
        if val > 0.5:
            print(f'  Start at {t}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')