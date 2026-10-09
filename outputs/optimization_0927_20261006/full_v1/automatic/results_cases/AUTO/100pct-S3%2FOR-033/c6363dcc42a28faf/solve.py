import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df['Time'])
if len(period_ids) != 48:
    raise ValueError(f'Expected 48 periods, got {len(period_ids)}')
period_id_to_idx = {pid: idx for (idx, pid) in enumerate(period_ids)}
period_idx_to_id = {idx: pid for (idx, pid) in enumerate(period_ids)}
try:
    requirement_per_period = df['Requirement'].astype(int).to_list()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
if len(requirement_per_period) != 48:
    raise ValueError(f'Expected 48 requirements, got {len(requirement_per_period)}')
num_periods = 48
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(range(num_periods), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in range(num_periods))), gp.GRB.MINIMIZE)
for t in range(num_periods):
    covering_starts = []
    for s in range(num_periods):
        covered_periods = [(s + offset) % num_periods for offset in range(shift_length)]
        if t in covered_periods:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift covers period {t} ({period_idx_to_id[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement_per_period[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in range(num_periods):
        num_staff = int(round(x_vars[s].X))
        if num_staff > 0:
            print(f"Start at '{period_idx_to_id[s]}': {num_staff} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')