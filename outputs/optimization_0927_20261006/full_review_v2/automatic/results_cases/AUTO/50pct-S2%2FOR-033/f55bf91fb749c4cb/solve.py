import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
time_slots = list(df['Time'])
if len(time_slots) != 48:
    raise ValueError(f'Expected 48 time slots, got {len(time_slots)}.')
slot_idx_to_label = {i: time_slots[i] for i in range(48)}
slot_label_to_idx = {label: i for (i, label) in slot_idx_to_label.items()}
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
requirement_by_idx = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
num_slots = 48
shift_length = 16
shift_start_indices = list(range(num_slots))
shift_coverage = dict()
for s in shift_start_indices:
    covered = [(s + k) % num_slots for k in range(shift_length)]
    shift_coverage[s] = set(covered)
slot_covered_by_shifts = dict()
for t in range(num_slots):
    slot_covered_by_shifts[t] = set()
    for s in shift_start_indices:
        if t in shift_coverage[s]:
            slot_covered_by_shifts[t].add(s)
    if len(slot_covered_by_shifts[t]) == 0:
        raise ValueError(f'Time slot {t} ({slot_idx_to_label[t]}) is not covered by any shift.')
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
for t in range(num_slots):
    m.addConstr(gp.quicksum((x_vars[s] for s in slot_covered_by_shifts[t])) >= requirement_by_idx[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        val = x_vars[s].X
        if val > 1e-06:
            print(f'  Start at {slot_idx_to_label[s]}: {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')