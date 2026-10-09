import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = df['Time'].tolist()
num_periods = len(periods)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
shift_starts = periods.copy()
num_shifts = num_periods
requirement = {}
for (idx, row) in df.iterrows():
    period = row['Time']
    try:
        req = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement value at period '{period}': {row['Requirement']}") from e
    requirement[period] = req
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
period_idx = {period: idx for (idx, period) in enumerate(periods)}
for (t_idx, t) in enumerate(periods):
    covering_shifts = []
    for (s_idx, s) in enumerate(shift_starts):
        covered_indices = [(s_idx + offset) % num_periods for offset in range(16)]
        if t_idx in covered_indices:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f"No shift covers period '{t}' (index {t_idx})")
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t_idx}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Assignments ---')
    for s in shift_starts:
        val = x_vars[s].X
        if val > 1e-06:
            print(f"Shift starting at '{s}': {int(round(val))} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')