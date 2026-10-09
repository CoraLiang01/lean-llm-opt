import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
time_slots = list(df.index)
n_slots = len(time_slots)
if n_slots != 48:
    raise ValueError(f'Expected 48 time slots, got {n_slots}')
shift_starts = list(df.index)
slot_labels = df['Time'].tolist()
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
shift_length = 16
for t in time_slots:
    covering_shifts = []
    for s in shift_starts:
        covered_slots = [(s + i) % n_slots for i in range(shift_length)]
        if t in covered_slots:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shift covers time slot {t} ({slot_labels[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_starts:
        n_staff = x_vars[s].X
        if n_staff > 0.5:
            print(f'Shift start at slot {s} ({slot_labels[s]}): {int(round(n_staff))} staff')
    print('\n--- Coverage per Time Slot ---')
    for t in time_slots:
        covered = sum((x_vars[s].X for s in shift_starts if t in [(s + i) % n_slots for i in range(shift_length)]))
        print(f'Time slot {t} ({slot_labels[t]}): Required={requirement[t]}, Covered={int(round(covered))}')
else:
    print(f'No optimal solution found. Status: {m.status}')