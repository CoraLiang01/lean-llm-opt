import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
time_slots = list(df['Time'].astype(str))
n_slots = len(time_slots)
if n_slots != 48:
    raise ValueError(f'Expected 48 time slots, got {n_slots}')
slot_idx_to_label = {i: time_slots[i] for i in range(n_slots)}
slot_label_to_idx = {label: i for (i, label) in slot_idx_to_label.items()}
requirements = df['Requirement'].apply(lambda x: int(x.strip())).to_numpy()
if requirements.shape[0] != n_slots:
    raise ValueError('Mismatch in number of requirements and time slots.')
shift_length = 16
shift_start_indices = list(range(n_slots))
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in range(n_slots):
    covering_shift_starts = []
    for s in shift_start_indices:
        if 0 <= (t - s) % n_slots < shift_length:
            covering_shift_starts.append(s)
    if not covering_shift_starts:
        raise ValueError(f'No shift covers time slot {t} ({slot_idx_to_label[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shift_starts)) >= requirements[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        n_staff = x_vars[s].X
        if n_staff > 1e-06:
            print(f"  Shift starting at '{slot_idx_to_label[s]}': {int(round(n_staff))} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')