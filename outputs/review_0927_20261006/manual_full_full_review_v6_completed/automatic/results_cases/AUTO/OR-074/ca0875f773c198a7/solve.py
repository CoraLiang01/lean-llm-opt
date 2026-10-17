import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and 'Requirement' not found in CSV.")
periods = df['Time'].astype(str).str.strip().tolist()
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}.')
period_idx_to_label = {i: periods[i] for i in range(48)}
period_label_to_idx = {label: i for i, label in period_idx_to_label.items()}
try:
    requirements = df['Requirement'].astype(int).tolist()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
if len(requirements) != 48:
    raise ValueError(f'Expected 48 requirements, got {len(requirements)}.')
m = gp.Model('WaitstaffScheduling')
shift_start_indices = list(range(48))
shift_start_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
shift_length = 16
for t in range(48):
    covering_shift_starts = []
    for s in shift_start_indices:
        covered_periods = [(s + offset) % 48 for offset in range(shift_length)]
        if t in covered_periods:
            covering_shift_starts.append(s)
    if not covering_shift_starts:
        raise ValueError(f'No shift covers period {t} ({period_idx_to_label[t]})')
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_shift_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        num_staff = int(round(shift_start_vars[s].X))
        if num_staff > 0:
            print(f"  Shift starting at '{period_idx_to_label[s]}': {num_staff} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')