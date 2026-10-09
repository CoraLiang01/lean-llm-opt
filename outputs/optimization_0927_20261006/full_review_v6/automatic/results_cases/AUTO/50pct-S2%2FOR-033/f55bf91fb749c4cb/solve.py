import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and/or 'Requirement' not found in CSV.")
periods = df['Time'].astype(str).str.strip().tolist()
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}.')
period_idx = {p: i for (i, p) in enumerate(periods)}
idx_period = {i: p for (i, p) in enumerate(periods)}
requirements = {}
for (i, row) in df.iterrows():
    period = str(row['Time']).strip()
    try:
        req = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement value at period '{period}': {row['Requirement']}")
    requirements[period] = req
S = periods
T = periods
shift_length = 16
num_periods = len(periods)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(S, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in T:
    t_idx = period_idx[t]
    covering_shifts = []
    for s in S:
        s_idx = period_idx[s]
        covered_indices = [(s_idx + offset) % num_periods for offset in range(shift_length)]
        if t_idx in covered_indices:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f"No shift covers period '{t}' (index {t_idx})")
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t_idx}')
m.setObjective(gp.quicksum((x_vars[s] for s in S)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in S:
        val = x_vars[s].X
        if val > 0.5:
            print(f"  Start at '{s}': {int(round(val))} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')