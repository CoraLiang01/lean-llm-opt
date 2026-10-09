import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and/or 'Requirement' not found in CSV.")
period_ids = list(df['Time'])
n_periods = len(period_ids)
if n_periods != 48:
    raise ValueError(f'Expected 48 time periods, got {n_periods}.')
period_id_to_idx = {pid: idx for (idx, pid) in enumerate(period_ids)}
period_idx_to_id = {idx: pid for (idx, pid) in enumerate(period_ids)}
try:
    requirement = df['Requirement'].astype(int).to_list()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
shift_length = 16
shift_start_indices = list(range(n_periods))
shift_covers = dict()
for s in shift_start_indices:
    covered = [(s + offset) % n_periods for offset in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = dict()
for t in range(n_periods):
    covering_shifts = []
    for s in shift_start_indices:
        if t in shift_covers[s]:
            covering_shifts.append(s)
    period_covered_by[t] = covering_shifts
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
for t in range(n_periods):
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by[t])) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x_vars[s].X for s in shift_start_indices))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        num_staff = int(round(x_vars[s].X))
        if num_staff > 0:
            shift_start_label = period_idx_to_id[s]
            shift_end_idx = (s + shift_length - 1) % n_periods
            shift_end_label = period_idx_to_id[shift_end_idx]
            print(f"  {num_staff} staff start at '{shift_start_label}' (covering {shift_length} periods through '{shift_end_label}')")
else:
    print(f'No optimal solution found. Status: {m.status}')