import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
num_slots = df.shape[0]
if num_slots != 48:
    raise ValueError(f'Expected 48 time slots, got {num_slots}')
slot_idx_to_label = df['Time'].tolist()
try:
    requirement_per_slot = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
shift_length = 16
slots = list(range(num_slots))
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(slots, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in slots)), gp.GRB.MINIMIZE)
for t in slots:
    covering_starts = [s for s in slots if (t - s) % num_slots < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement_per_slot[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in slots:
        n = int(round(x_vars[s].X))
        if n > 0:
            shift_start_label = slot_idx_to_label[s]
            shift_end_idx = (s + shift_length) % num_slots
            shift_end_label = slot_idx_to_label[shift_end_idx]
            print(f'  Start at slot {s:2d} ({shift_start_label}): {n} staff (covers to {shift_end_label})')
else:
    print(f'No optimal solution found. Status: {m.status}')