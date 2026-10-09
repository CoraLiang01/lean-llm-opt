import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df['Time'])
if len(period_ids) != 48:
    raise ValueError(f'Expected 48 periods, got {len(period_ids)}')
period_idx_to_id = {i: period_ids[i] for i in range(48)}
period_id_to_idx = {period_ids[i]: i for i in range(48)}
try:
    requirement_per_period = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
requirement = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
m = gp.Model('WaitstaffScheduling')
shift_start_indices = list(range(48))
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
shift_length = 16
for t in range(48):
    covering_starts = []
    for s in range(48):
        if 0 <= (t - s) % 48 < shift_length:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift starts cover period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        num = x_vars[s].X
        if num > 0.5:
            print(f'  Start at period {s} ({period_idx_to_id[s]}): {int(round(num))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')