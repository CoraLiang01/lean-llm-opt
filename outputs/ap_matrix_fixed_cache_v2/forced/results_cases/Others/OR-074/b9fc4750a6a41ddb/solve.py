import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and 'Requirement' not found in CSV.")
df['Time'] = df['Time'].str.strip()
time_periods = list(df['Time'])
if len(time_periods) != 48:
    raise ValueError(f'Expected 48 time periods, got {len(time_periods)}.')
try:
    requirements = {row['Time']: int(row['Requirement']) for _, row in df.iterrows()}
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' to int: {e}")
shift_starts = list(time_periods)
num_periods = len(time_periods)
period_idx = {label: idx for idx, label in enumerate(time_periods)}
idx_to_label = {idx: label for idx, label in enumerate(time_periods)}
shift_coverage = dict()
for s_idx in range(num_periods):
    covered = set(((s_idx + offset) % num_periods for offset in range(16)))
    shift_coverage[s_idx] = covered
period_covered_by = dict()
for t_idx in range(num_periods):
    covering = set()
    for s_idx in range(num_periods):
        if t_idx in shift_coverage[s_idx]:
            covering.add(s_idx)
    period_covered_by[t_idx] = covering
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(num_periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s_idx] for s_idx in range(num_periods))), gp.GRB.MINIMIZE)
for t_idx, t_label in idx_to_label.items():
    required = requirements[t_label]
    covering_s = period_covered_by[t_idx]
    m.addConstr(gp.quicksum((x_vars[s_idx] for s_idx in covering_s)) >= required, name=f'cover_{t_idx}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x_vars[s_idx].X for s_idx in range(num_periods)))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s_idx, s_label in idx_to_label.items():
        val = x_vars[s_idx].X
        if val > 1e-06:
            print(f'  Shift starting at {s_label}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')